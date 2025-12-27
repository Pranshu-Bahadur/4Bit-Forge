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

    deq_out   = __fmaf_rn(static_cast<float>(q), s, -q0 * s);
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
// Kernel 1: Quantize block (row-parallel, no delta zeroing)
// grid.x = B (rows in block), blockDim.x = threads_quant
// ----------------------------------------------------------------------------

template <typename scalar_t, int WARPS_PER_BLOCK, bool VECTORIZED>
__global__ void gptq_quant_block_kernel(
    scalar_t* __restrict__ W,             // [C, R]
    uint8_t* __restrict__ qweight,        // [C, R]
    const float* __restrict__ scales,     // [R*G]
    const float* __restrict__ qzeros,     // [R*G]
    const int32_t* __restrict__ g_idx,    // [C] (optional), maps j -> group
    float* __restrict__ delta_block,      // [B, R]
    int64_t C, int64_t R, int64_t G,
    int64_t block_start, int64_t block_end,
    int64_t group_size,
    uint8_t bits
) {
    const int B = (int)(block_end - block_start);
    if (B <= 0) return;

    const int row_in_block = (int)blockIdx.x;
    if (row_in_block >= B) return;

    const int tid = (int)threadIdx.x;
    const int j   = (int)block_start + row_in_block;   // C-index (column being quantized)
    if (j < 0 || j >= (int)C) return;

    scalar_t* W_row = W + (int64_t)j * R;
    uint8_t*  Q_row = qweight + (int64_t)j * R;
    float*    D_row = delta_block + (int64_t)row_in_block * R;

    // group id is a function of j (C-axis), not col (R-axis)
    int g = g_idx ? (int)g_idx[j] : (j / (int)group_size);
    if (g < 0) g = 0;
    if (g >= (int)G) g = (int)G - 1;

    const int maxq_i = (1 << bits) - 1;

    if constexpr (VECTORIZED) {
        // NOTE: with scales/qzeros laid out as [r*G + g], each of the 4 rows
        // generally has DIFFERENT (s,q0). So you can't use a single (s,q0) for all 4.
        // Easiest bring-up: set VECTORIZED=false until you rework the 4-pack path.

        const int R_vec = (int)R / 4;
        auto* Q_row_vec = reinterpret_cast<uint32_t*>(Q_row);

        for (int col_v = tid; col_v < R_vec; col_v += blockDim.x) {
            const int r0 = col_v * 4 + 0;
            const int r1 = col_v * 4 + 1;
            const int r2 = col_v * 4 + 2;
            const int r3 = col_v * 4 + 3;

            float x0 = (float)W_row[r0];
            float x1 = (float)W_row[r1];
            float x2 = (float)W_row[r2];
            float x3 = (float)W_row[r3];

            // per-row meta: idx = r*G + g
            float s0  = scales[r0 * (int)G + g], q00 = qzeros[r0 * (int)G + g];
            float s1  = scales[r1 * (int)G + g], q01 = qzeros[r1 * (int)G + g];
            float s2  = scales[r2 * (int)G + g], q02 = qzeros[r2 * (int)G + g];
            float s3  = scales[r3 * (int)G + g], q03 = qzeros[r3 * (int)G + g];

            float inv0 = 1.0f / s0, inv1 = 1.0f / s1, inv2 = 1.0f / s2, inv3 = 1.0f / s3;

            float d0, deq0; uint8_t q0;
            float d1, deq1; uint8_t q1;
            float d2, deq2; uint8_t q2;
            float d3, deq3; uint8_t q3;

            quantize_scalar(x0, inv0, s0, q00, maxq_i, d0, q0, deq0);
            quantize_scalar(x1, inv1, s1, q01, maxq_i, d1, q1, deq1);
            quantize_scalar(x2, inv2, s2, q02, maxq_i, d2, q2, deq2);
            quantize_scalar(x3, inv3, s3, q03, maxq_i, d3, q3, deq3);

            W_row[r0] = (scalar_t)deq0; D_row[r0] = d0;
            W_row[r1] = (scalar_t)deq1; D_row[r1] = d1;
            W_row[r2] = (scalar_t)deq2; D_row[r2] = d2;
            W_row[r3] = (scalar_t)deq3; D_row[r3] = d3;

            // pack 4x 8-bit into u32
            Q_row_vec[col_v] = (uint32_t)q0 | ((uint32_t)q1 << 8) | ((uint32_t)q2 << 16) | ((uint32_t)q3 << 24);
        }
    } else {
        for (int col = tid; col < (int)R; col += blockDim.x) {
            // idx = r*G + g
            const int idx = col * (int)G + g;
            float s  = scales[idx];
            float q0 = qzeros[idx];
            float inv_s = 1.0f / s;

            float delta, deq;
            uint8_t q_byte;

            quantize_scalar((float)W_row[col], inv_s, s, q0, maxq_i, delta, q_byte, deq);

            W_row[col] = (scalar_t)deq;
            Q_row[col] = q_byte;
            D_row[col] = delta;
        }
    }
}


// ----------------------------------------------------------------------------
// Kernel 2: TRSM (Solve A * X = B)  with A lower-triangular
// A: [B, B], Bmat: [N, B], Xmat: [N, B]
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

    #pragma unroll
    for (int i = 0; i < MAX_B; ++i) {
        x[i] = 0.f;
        b[i] = 0.f;
    }

    for (int i = 0; i < B; ++i) {
        b[i] = Bmat[row * B + i];
    }

    // Forward substitution
    for (int i = 0; i < B; ++i) {
        float sum = 0.f;
        for (int k = 0; k < i; ++k) {
            sum += A[i * B + k] * x[k];
        }
        float diag = A[i * B + i];
        float rhs  = b[i] - sum;
        x[i] = rhs / diag;
    }

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
    int r = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int c = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (r >= R || c >= C_tail) return;

    float acc = 0.0f;
    for (int k = 0; k < B; ++k) {
        float h_val = H[k * stride_h_b + c * stride_h_c];
        float e_val = E[k * stride_e_b + r * stride_e_r];
        acc += h_val * e_val;
    }

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
    torch::Tensor scales, 
    torch::Tensor qzeros,
    int64_t group_size,
    int64_t bits,
    int64_t block_size,
    torch::Tensor g_idx,
    int64_t G
) {
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(hessian_inv.is_cuda(), "hessian_inv must be CUDA");
    
    weight      = weight.contiguous();
    hessian_inv = hessian_inv.contiguous();
    scales = scales.contiguous();
    qzeros = qzeros.contiguous();

    const int64_t C = weight.size(0);
    const int64_t R = weight.size(1);

    // Clamp block size
    constexpr int MAX_BLOCK_SIZE = 32;
    if (block_size <= 0 || block_size > MAX_BLOCK_SIZE) block_size = MAX_BLOCK_SIZE;
    if (block_size > C) block_size = C;

    auto qweight     = torch::empty({C, R}, torch::TensorOptions().dtype(torch::kUInt8).device(weight.device()));
    auto delta_block = torch::empty({block_size, R}, torch::TensorOptions().dtype(at::kFloat).device(weight.device()));

    

    auto stream = at::cuda::getCurrentCUDAStream();
    const bool use_vectorized = (R % 4 == 0) && (group_size % 4 == 0);
    constexpr int WARPS_PER_BLOCK = 8;
    const int threads_quant = WARPS_PER_BLOCK * 32;

    // Macro for dispatching quant kernel
    #define LAUNCH_QKERNEL(SCALAR_T) \
        if (use_vectorized) { \
            gptq_quant_block_kernel<SCALAR_T, WARPS_PER_BLOCK, true><<< \
                B, threads_quant, 0, stream>>>( \
                weight.data_ptr<SCALAR_T>(), \
                qweight.data_ptr<uint8_t>(), \
                scales.data_ptr<float>(), \
                qzeros.data_ptr<float>(), \
                g_idx.data_ptr<int32_t>(), \
                delta_block.data_ptr<float>(), \
                C, R, G, \
                block_start, block_end, \
                group_size, \
                static_cast<uint8_t>(bits)); \
        } else { \
            gptq_quant_block_kernel<SCALAR_T, WARPS_PER_BLOCK, false><<< \
                B, threads_quant, 0, stream>>>( \
                weight.data_ptr<SCALAR_T>(), \
                qweight.data_ptr<uint8_t>(), \
                scales.data_ptr<float>(), \
                qzeros.data_ptr<float>(), \
                g_idx.data_ptr<int32_t>(), \
                delta_block.data_ptr<float>(), \
                C, R, G, \
                block_start, block_end, \
                group_size, \
                static_cast<uint8_t>(bits)); \
        }

    // --- Main Block Loop ---
    for (int64_t block_start = 0; block_start < C; block_start += block_size) {
        const int64_t block_end = std::min(block_start + block_size, C);
        const int64_t B_long    = block_end - block_start;
        const int B             = static_cast<int>(B_long);
        const int N             = static_cast<int>(R);

        // 1. Quantize block (row-parallel)
        AT_DISPATCH_SWITCH(weight.scalar_type(), "gptq_quant_block",
            AT_DISPATCH_CASE(at::ScalarType::Float,         [&] { LAUNCH_QKERNEL(float); })
            AT_DISPATCH_CASE(at::ScalarType::Half,          [&] { LAUNCH_QKERNEL(at::Half); })
            AT_DISPATCH_CASE(at::ScalarType::BFloat16,      [&] { LAUNCH_QKERNEL(at::BFloat16); })
            AT_DISPATCH_CASE(at::ScalarType::Float8_e4m3fn, [&] { LAUNCH_QKERNEL(c10::Float8_e4m3fn); })
        );
        CUDA_CHECK(cudaGetLastError());

        if (block_end >= C) continue;

        // 2. Prepare TRSM inputs
        auto H_block = hessian_inv.narrow(0, block_start, B_long)
                                  .narrow(1, block_start, B_long)
                                  .to(torch::kFloat);                // [B, B]
        auto A_lower = H_block.t().contiguous();                     // [B, B] lower-triangular

        auto Delta_J = delta_block.narrow(0, 0, B_long);             // [B, R]
        auto Delta_T = Delta_J.t().contiguous();                     // [R, B]

        // 3. Solve A_lower * E_T = Delta_T  => E_T: [R, B]
        auto E_T = torch::empty_like(Delta_T);                       // [R, B]
        {
            const int threads_trsm = 256;
            const int blocks_trsm  = (N + threads_trsm - 1) / threads_trsm;
            block_trsm_lower_kernel<MAX_BLOCK_SIZE><<<
                blocks_trsm, threads_trsm, 0, stream>>>(
                    A_lower.data_ptr<float>(),
                    Delta_T.data_ptr<float>(),
                    E_T.data_ptr<float>(),
                    B, N
            );
        }
        CUDA_CHECK(cudaGetLastError());

        // 4. Tail update: W_tail -= H_cross^T @ E_J
        auto H_cross = hessian_inv.narrow(0, block_start, B_long)
                                  .narrow(1, block_end, C - block_end);   // [B, C_tail]

        auto W_tail = weight.narrow(0, block_end, C - block_end);         // [C_tail, R]
        auto E_J    = E_T.t();                                            // [B, R]

        if (weight.scalar_type() == at::ScalarType::Float8_e4m3fn) {
            auto H_cross_f   = H_cross.to(torch::kFloat).contiguous(); 
            auto E_J_contig  = E_J.contiguous();

            int C_tail_int = static_cast<int>(W_tail.size(0));
            int R_int      = static_cast<int>(W_tail.size(1));

            dim3 block(16, 16);
            dim3 grid((R_int + 15) / 16, (C_tail_int + 15) / 16);

            update_tail_kernel<c10::Float8_e4m3fn><<<grid, block, 0, stream>>>(
                W_tail.data_ptr<c10::Float8_e4m3fn>(),
                H_cross_f.data_ptr<float>(),
                E_J_contig.data_ptr<float>(),
                C_tail_int, R_int, B,
                static_cast<int>(W_tail.stride(0)),
                static_cast<int>(W_tail.stride(1)),
                static_cast<int>(H_cross_f.stride(0)),
                static_cast<int>(H_cross_f.stride(1)),
                static_cast<int>(E_J_contig.stride(0)),
                static_cast<int>(E_J_contig.stride(1))
            );
            CUDA_CHECK(cudaGetLastError());
        } else {
            // Standard high-precision path (fp16/bf16/fp32): W_tail -= H_cross^T @ E_J
            W_tail.addmm_(
                H_cross.t(),
                E_J.to(weight.scalar_type()),
                -1.0f, 1.0f
            );
        }
    }

    return qweight;
}
