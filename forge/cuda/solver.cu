#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>

// Headers for specific types
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t _err = (expr);                                                \
    TORCH_CHECK(_err == cudaSuccess, "CUDA error: ",                          \
                cudaGetErrorString(_err), " at ", __FILE__, ":", __LINE__);   \
  } while (0)

namespace {

// --- Data Structures --------------------------------------------------------

struct QMetaPacked {
    int16_t  log2_scale_fp;  // Q8.8 fixed-point log2(scale)
    uint8_t  qzero;          // zero-point (0..255)
    uint8_t  flags;          // bitfield; bit0 = symmetric
};
static_assert(sizeof(QMetaPacked) == 4, "QMetaPacked must be 4 bytes.");

__device__ __forceinline__ uint32_t qmeta_to_u32(const QMetaPacked& m) {
    return *reinterpret_cast<const uint32_t*>(&m);
}

// --- Generic Load/Store Helpers (Vectorized) -------------------------------

template <typename T>
__device__ __forceinline__ void load_4(const T* src, float* dst) {
    if constexpr (sizeof(T) == 4) {
        float4 v = *reinterpret_cast<const float4*>(src);
        dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
    } 
    else if constexpr (sizeof(T) == 2) {
        int2 v = *reinterpret_cast<const int2*>(src);
        const T* vals = reinterpret_cast<const T*>(&v);
        dst[0] = static_cast<float>(vals[0]);
        dst[1] = static_cast<float>(vals[1]);
        dst[2] = static_cast<float>(vals[2]);
        dst[3] = static_cast<float>(vals[3]);
    } 
    else if constexpr (sizeof(T) == 1) {
        int v = *reinterpret_cast<const int*>(src);
        const T* vals = reinterpret_cast<const T*>(&v);
        dst[0] = static_cast<float>(vals[0]);
        dst[1] = static_cast<float>(vals[1]);
        dst[2] = static_cast<float>(vals[2]);
        dst[3] = static_cast<float>(vals[3]);
    }
}

template <typename T>
__device__ __forceinline__ void store_4(T* dst, const float* src) {
    if constexpr (sizeof(T) == 4) {
        float4 v;
        v.x = src[0]; v.y = src[1]; v.z = src[2]; v.w = src[3];
        *reinterpret_cast<float4*>(dst) = v;
    } 
    else if constexpr (sizeof(T) == 2) {
        int2 v;
        T* vals = reinterpret_cast<T*>(&v);
        vals[0] = static_cast<T>(src[0]);
        vals[1] = static_cast<T>(src[1]);
        vals[2] = static_cast<T>(src[2]);
        vals[3] = static_cast<T>(src[3]);
        *reinterpret_cast<int2*>(dst) = v;
    } 
    else if constexpr (sizeof(T) == 1) {
        int v;
        T* vals = reinterpret_cast<T*>(&v);
        vals[0] = static_cast<T>(src[0]);
        vals[1] = static_cast<T>(src[1]);
        vals[2] = static_cast<T>(src[2]);
        vals[3] = static_cast<T>(src[3]);
        *reinterpret_cast<int*>(dst) = v;
    }
}

// --- QMeta Decoding ---------------------------------------------------------

__device__ __forceinline__ void decode_qmeta(
    uint32_t packed,
    float&   scale,
    float&   inv_scale,
    float&   qzero_f,
    float&   maxq_g,
    uint8_t  global_bits
) {
    int16_t log2_q88 = static_cast<int16_t>(packed & 0xFFFFu);
    uint8_t qzero_u8 = static_cast<uint8_t>((packed >> 16) & 0xFFu);
    uint8_t flags    = static_cast<uint8_t>(packed >> 24);

    constexpr float INV256 = 1.0f / 256.0f;
    float log2_scale = __int2float_rn(log2_q88) * INV256;
    scale     = exp2f(log2_scale);
    inv_scale = exp2f(-log2_scale);

    int maxq_i = (1 << global_bits) - 1;
    maxq_g = static_cast<float>(maxq_i);

    if (flags & 0x01) {
        constexpr float HALF = 0.5f;
        qzero_u8 = static_cast<uint8_t>((maxq_g + 1.0f) * HALF);
    }
    qzero_f = static_cast<float>(qzero_u8);
}

// --- Quantization Primitives ------------------------------------------------

__device__ __forceinline__ void quantize_scalar(
    float x, float inv_s, float s, float q0, int maxq_i,
    float& delta_out, uint8_t& q_out, float& deq_out
) {
    float biased = x * inv_s + q0;
    int q = __float2int_rn(biased);

    if (q < 0) q = 0;
    else if (q > maxq_i) q = maxq_i;

    deq_out    = __fmaf_rn(static_cast<float>(q), s, -q0 * s);
    delta_out = x - deq_out;       // W_old - Q
    q_out     = static_cast<uint8_t>(q);
}

__device__ __forceinline__ void quantize_process_4(
    const float* x_vals,
    float inv_s, float s, float q0, int maxq_i,
    float* delta_vals, uint32_t& q_packed, float* deq_vals
) {
    uint8_t q_b[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        quantize_scalar(x_vals[i], inv_s, s, q0, maxq_i,
                        delta_vals[i], q_b[i], deq_vals[i]);
    }
    q_packed =  (uint32_t)q_b[0]
              | ((uint32_t)q_b[1] << 8)
              | ((uint32_t)q_b[2] << 16)
              | ((uint32_t)q_b[3] << 24);
}

// ----------------------------------------------------------------------------
// Kernel 1: Quantize block
// ----------------------------------------------------------------------------

template <typename scalar_t, int WARPS_PER_BLOCK, bool VECTORIZED>
__global__ void gptq_quant_block_kernel(
    scalar_t* __restrict__ W,             // [C, R]
    uint8_t* __restrict__ qweight,        // [C, R]
    const QMetaPacked* __restrict__ qmeta,// [C * G]
    float* __restrict__ delta_block,      // [B, R]
    int64_t C, int64_t R, int64_t G,
    int64_t block_start, int64_t block_end,
    int64_t group_size,
    uint8_t bits
) {
    const int B = static_cast<int>(block_end - block_start);
    if (B <= 0) return;

    const int tid = threadIdx.x;

    // Clear delta buffer
    if (VECTORIZED) {
        const int total_vecs = (B * static_cast<int>(R)) / 4;
        float4* delta_vec = reinterpret_cast<float4*>(delta_block);
        for (int idx = tid; idx < total_vecs; idx += blockDim.x) {
            delta_vec[idx] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    } else {
        const int total_elems = B * static_cast<int>(R);
        for (int idx = tid; idx < total_elems; idx += blockDim.x) {
            delta_block[idx] = 0.f;
        }
    }
    __syncthreads();

    // Quantize rows
    for (int j = static_cast<int>(block_start); j < static_cast<int>(block_end); ++j) {
        const int row_in_block = j - static_cast<int>(block_start);

        scalar_t* W_row = W + j * R;
        uint8_t* Q_row = qweight + j * R;
        float* D_row = delta_block + row_in_block * R;

        if (VECTORIZED) {
            const int R_vec          = static_cast<int>(R) / 4;
            const int group_size_vec = static_cast<int>(group_size) / 4;
            auto* Q_row_vec          = reinterpret_cast<uint32_t*>(Q_row);
            auto* D_row_vec          = reinterpret_cast<float4*>(D_row);

            for (int col_v = tid; col_v < R_vec; col_v += blockDim.x) {
                const int g = col_v / group_size_vec;
                const QMetaPacked m = qmeta[j * static_cast<int>(G) + g];
                float s, inv_s, q0, maxq_g;
                decode_qmeta(qmeta_to_u32(m), s, inv_s, q0, maxq_g, bits);
                
                float x_vals[4];
                load_4(W_row + col_v * 4, x_vals);

                float delta_vals[4], deq_vals[4];
                uint32_t q_packed;

                quantize_process_4(x_vals, inv_s, s, q0, static_cast<int>(maxq_g),
                                   delta_vals, q_packed, deq_vals);

                store_4(W_row + col_v * 4, deq_vals);
                Q_row_vec[col_v] = q_packed;
                D_row_vec[col_v] = make_float4(delta_vals[0], delta_vals[1], delta_vals[2], delta_vals[3]);
            }
        } else {
            for (int col = tid; col < static_cast<int>(R); col += blockDim.x) {
                const int g = col / static_cast<int>(group_size);
                const QMetaPacked m = qmeta[j * static_cast<int>(G) + g];
                float s, inv_s, q0, maxq_g;
                decode_qmeta(qmeta_to_u32(m), s, inv_s, q0, maxq_g, bits);

                float delta, deq;
                uint8_t q_byte;
                quantize_scalar(static_cast<float>(W_row[col]), inv_s, s, q0, static_cast<int>(maxq_g),
                                delta, q_byte, deq);

                W_row[col] = static_cast<scalar_t>(deq);
                Q_row[col] = q_byte;
                D_row[col] = delta;
            }
        }
        __syncthreads();
    }
}

// ----------------------------------------------------------------------------
// Kernel 2: TRSM (Solve A * X = B)
// ----------------------------------------------------------------------------

template <int MAX_B>
__global__ void block_trsm_lower_kernel(
    const float* __restrict__ A,    // [B, B]
    const float* __restrict__ Bmat, // [N, B]
    float* __restrict__ Xmat,       // [N, B]
    int B, int N
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    float x[MAX_B];
    float b[MAX_B];

    // Initialize
    #pragma unroll
    for (int i = 0; i < MAX_B; ++i) {
        x[i] = 0.f;
        b[i] = 0.f;
    }

    // Load RHS
    for (int i = 0; i < B; ++i) {
        b[i] = Bmat[row * B + i];
    }

    // Forward sub
    for (int i = 0; i < B; ++i) {
        float sum = 0.f;
        for (int k = 0; k < i; ++k) {
            sum += A[i * B + k] * x[k];
        }
        float diag = A[i * B + i];
        float rhs  = b[i] - sum;
        x[i] = rhs / diag;
    }

    // Store
    for (int i = 0; i < B; ++i) {
        Xmat[row * B + i] = x[i];
    }
}

// ----------------------------------------------------------------------------
// Kernel 3: Custom Tail Update for FP8 Support
// W_tail -= H_cross^T @ E_J
// 
// W_tail: [C_tail, R] (scalar_t, e.g. fp8)
// H_cross:[B, C_tail] (float, pre-casted for precision/convenience)
// E_J:    [B, R]      (float)
// ----------------------------------------------------------------------------

template <typename scalar_t>
__global__ void update_tail_kernel(
    scalar_t* __restrict__ W,    // [C_tail, R]
    const float* __restrict__ H, // [B, C_tail]
    const float* __restrict__ E, // [B, R]
    int C_tail, int R, int B,
    int stride_w_c, int stride_w_r,
    int stride_h_b, int stride_h_c,
    int stride_e_b, int stride_e_r
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x; // Col index (0..R)
    int c = blockIdx.y * blockDim.y + threadIdx.y; // Row index (0..C_tail)

    if (r >= R || c >= C_tail) return;

    float acc = 0.0f;
    // Dot product: H_cross^T[c, :] . E_J[:, r]
    // = H_cross[:, c] . E_J[:, r]
    // = sum_k ( H[k, c] * E[k, r] )
    for (int k = 0; k < B; ++k) {
        float h_val = H[k * stride_h_b + c * stride_h_c]; 
        float e_val = E[k * stride_e_b + r * stride_e_r];
        acc += h_val * e_val;
    }

    // W[c, r] -= acc
    int w_idx = c * stride_w_c + r * stride_w_r;
    float w_val = static_cast<float>(W[w_idx]);
    w_val -= acc;
    W[w_idx] = static_cast<scalar_t>(w_val);
}

} // anonymous namespace

// ----------------------------------------------------------------------------
// Host Wrapper
// ----------------------------------------------------------------------------

torch::Tensor gptq_solver_cuda(
    torch::Tensor weight,       // [C, R]
    torch::Tensor hessian_inv,  // [C, C]
    torch::Tensor qmeta_bytes,  // [C, G, 4]
    int64_t group_size,
    int64_t bits,
    int64_t block_size
) {
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(hessian_inv.is_cuda(), "hessian_inv must be CUDA");
    
    weight = weight.contiguous();
    // Do NOT cast hessian_inv globally.
    hessian_inv = hessian_inv.contiguous(); 
    qmeta_bytes = qmeta_bytes.contiguous();

    const int64_t C = weight.size(0);
    const int64_t R = weight.size(1);

    // Determine G
    int64_t G;
    if (qmeta_bytes.dim() == 3) {
        G = qmeta_bytes.size(1);
    } else {
        G = qmeta_bytes.size(0) / C;
    }

    // Clamp block size
    constexpr int64_t MAX_BLOCK_SIZE = 32;
    if (block_size <= 0 || block_size > MAX_BLOCK_SIZE) block_size = MAX_BLOCK_SIZE;
    if (block_size > C) block_size = C;

    auto qweight = torch::empty({C, R}, torch::TensorOptions().dtype(torch::kUInt8).device(weight.device()));
    auto delta_block = torch::empty({block_size, R}, torch::TensorOptions().dtype(at::kFloat).device(weight.device()));
    
    // Flatten metadata
    auto qmeta_flat = (qmeta_bytes.dim() == 3) ? qmeta_bytes.view({C * G, 4}) : qmeta_bytes;

    auto stream = at::cuda::getCurrentCUDAStream();
    const bool use_vectorized = (R % 4 == 0) && (group_size % 4 == 0);
    constexpr int WARPS_PER_BLOCK = 8;
    const int threads_quant = WARPS_PER_BLOCK * 32;

    // Macro for dispatching quant kernel
    #define LAUNCH_QKERNEL(SCALAR_T) \
        if (use_vectorized) { \
            gptq_quant_block_kernel<SCALAR_T, WARPS_PER_BLOCK, true><<<1, threads_quant, 0, stream>>>( \
                weight.data_ptr<SCALAR_T>(), qweight.data_ptr<uint8_t>(), \
                reinterpret_cast<const QMetaPacked*>(qmeta_flat.data_ptr<uint8_t>()), \
                delta_block.data_ptr<float>(), C, R, G, block_start, block_end, group_size, (uint8_t)bits); \
        } else { \
            gptq_quant_block_kernel<SCALAR_T, WARPS_PER_BLOCK, false><<<1, threads_quant, 0, stream>>>( \
                weight.data_ptr<SCALAR_T>(), qweight.data_ptr<uint8_t>(), \
                reinterpret_cast<const QMetaPacked*>(qmeta_flat.data_ptr<uint8_t>()), \
                delta_block.data_ptr<float>(), C, R, G, block_start, block_end, group_size, (uint8_t)bits); \
        }

    // --- Main Block Loop ---
    for (int64_t block_start = 0; block_start < C; block_start += block_size) {
        const int64_t block_end = std::min(block_start + block_size, C);
        const int64_t B_long = block_end - block_start;
        const int B = static_cast<int>(B_long);
        const int N = static_cast<int>(R);

        // 1. Quantize block
        AT_DISPATCH_SWITCH(weight.scalar_type(), "gptq_quant_block",
            AT_DISPATCH_CASE(at::ScalarType::Float,          [&] { LAUNCH_QKERNEL(float); })
            AT_DISPATCH_CASE(at::ScalarType::Half,           [&] { LAUNCH_QKERNEL(at::Half); })
            AT_DISPATCH_CASE(at::ScalarType::BFloat16,       [&] { LAUNCH_QKERNEL(at::BFloat16); })
            AT_DISPATCH_CASE(at::ScalarType::Float8_e4m3fn,  [&] { LAUNCH_QKERNEL(c10::Float8_e4m3fn); })
        );
        CUDA_CHECK(cudaGetLastError());

        if (block_end >= C) continue;

        // 2. Prepare TRSM inputs
        // Slice H[block, block], cast to float
        auto H_block = hessian_inv.narrow(0, block_start, B_long)
                                  .narrow(1, block_start, B_long)
                                  .to(torch::kFloat); // [B, B]
        auto A_lower = H_block.t().contiguous();      // Lower triangular

        auto Delta_J = delta_block.narrow(0, 0, B_long);     // [B, R]
        auto Delta_T = Delta_J.t().contiguous();             // [R, B]

        // 3. Solve A * E_T = Delta_T
        auto E_T = torch::empty_like(Delta_T); // [R, B]
        {
            const int threads_trsm = 256;
            const int blocks_trsm = (N + threads_trsm - 1) / threads_trsm;
            block_trsm_lower_kernel<MAX_BLOCK_SIZE><<<blocks_trsm, threads_trsm, 0, stream>>>(
                A_lower.data_ptr<float>(),
                Delta_T.data_ptr<float>(),
                E_T.data_ptr<float>(),
                B, N
            );
        }
        CUDA_CHECK(cudaGetLastError());

        // 4. Update Tail
        // Operation: W_tail -= H_cross.T @ E_J
        // H_cross: [B, C_tail]
        // E_J: [B, R]
        
        auto H_cross = hessian_inv.narrow(0, block_start, B_long)
                                  .narrow(1, block_end, C - block_end); // [B, C_tail]
        
        auto W_tail = weight.narrow(0, block_end, C - block_end); // [C_tail, R]
        auto E_J = E_T.t(); // [B, R]

        if (weight.scalar_type() == at::ScalarType::Float8_e4m3fn) {
            // --- Custom FP8 Update Path ---
            // Upcast small H_cross matrix to float for the kernel
            auto H_cross_f = H_cross.to(torch::kFloat).contiguous(); 
            auto E_J_contig = E_J.contiguous(); // Ensure E is contiguous

            int C_tail_int = static_cast<int>(W_tail.size(0));
            int R_int      = static_cast<int>(W_tail.size(1));

            dim3 block(16, 16);
            dim3 grid((R_int + 15) / 16, (C_tail_int + 15) / 16);

            update_tail_kernel<c10::Float8_e4m3fn><<<grid, block, 0, stream>>>(
                W_tail.data_ptr<c10::Float8_e4m3fn>(),
                H_cross_f.data_ptr<float>(),
                E_J_contig.data_ptr<float>(),
                C_tail_int, R_int, B,
                (int)W_tail.stride(0), (int)W_tail.stride(1),
                (int)H_cross_f.stride(0), (int)H_cross_f.stride(1),
                (int)E_J_contig.stride(0), (int)E_J_contig.stride(1)
            );
        } else {
            // --- Standard High-Performance Path (FP16/BF16/FP32) ---
            // W_tail.addmm_(mat1=H_cross.T, mat2=E_J, beta=1, alpha=-1)
            W_tail.addmm_(
                H_cross.t(), 
                E_J.to(weight.scalar_type()), 
                -1.0f, 1.0f
            );
        }
    }

    return qweight;
}