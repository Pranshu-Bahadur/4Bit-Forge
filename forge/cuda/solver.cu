#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>
#include <tuple>

// Headers for specific types
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

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
    uint8_t  flags;          // bitfield; bit0 = symmetric, others reserved
};
static_assert(sizeof(QMetaPacked) == 4, "QMetaPacked must be 4 bytes.");

// Helper: reinterpret QMetaPacked as uint32_t in device code
__device__ __forceinline__ uint32_t qmeta_to_u32(const QMetaPacked& m) {
    return *reinterpret_cast<const uint32_t*>(&m);
}

// --- Warp Utils -------------------------------------------------------------

// Broadcast a float from a specific lane to all lanes in the warp
__device__ __forceinline__ float warp_broadcast(float v, int src_lane = 0) {
    return __shfl_sync(0xffffffff, v, src_lane);
}

// --- Generic Load/Store Helpers (Vectorized) -------------------------------

// Loads 4 elements of type T from src into float dst[4]
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

// Stores 4 float values into dst as type T
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
    uint8_t  global_bits  // e.g. 4
) {
    int16_t log2_q88 = static_cast<int16_t>(packed & 0xFFFFu);
    uint8_t qzero_u8 = static_cast<uint8_t>((packed >> 16) & 0xFFu);
    uint8_t flags    = static_cast<uint8_t>(packed >> 24);

    constexpr float INV256 = 1.0f / 256.0f;
    float log2_scale = __int2float_rn(log2_q88) * INV256;
    scale      = exp2f(log2_scale);
    inv_scale = exp2f(-log2_scale); 

    uint8_t bits_g = global_bits;
    int maxq_i = (1 << bits_g) - 1;
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
    float& e, uint8_t& q_out, float& deq
) {
    float biased = x * inv_s + q0;
    int q = __float2int_rn(biased);
    
    if (q < 0) q = 0;
    else if (q > maxq_i) q = maxq_i;

    deq = __fmaf_rn(static_cast<float>(q), s, -q0 * s);
    e = deq - x;
    q_out = static_cast<uint8_t>(q);
}

__device__ __forceinline__ void quantize_process_4(
    const float* x_vals, 
    float inv_s, float s, float q0, int maxq_i,
    float* e_vals, uint32_t& q_packed, float* deq_vals
) {
    uint8_t q_b[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        quantize_scalar(x_vals[i], inv_s, s, q0, maxq_i, e_vals[i], q_b[i], deq_vals[i]);
    }
    q_packed = (uint32_t)q_b[0] | ((uint32_t)q_b[1] << 8) | ((uint32_t)q_b[2] << 16) | ((uint32_t)q_b[3] << 24);
}

// ----------------------------------------------------------------------------
// Main Kernel (No Shared Memory, Warp-Optimized)
// ----------------------------------------------------------------------------
template <typename scalar_t, int WARPS_PER_BLOCK, bool VECTORIZED>
__global__ void gptq_solver_kernel_no_smem(
    scalar_t* __restrict__ W,             
    uint8_t* __restrict__ qweight,        
    const scalar_t* __restrict__ Hinv,    
    const QMetaPacked* __restrict__ qmeta, 
    float* __restrict__ error_block,      
    int64_t C, int64_t R, int64_t G,
    int64_t block_start, int64_t block_end,
    int64_t group_size,
    uint8_t bits
) {
    const int B = static_cast<int>(block_end - block_start);
    if (B <= 0) return;

    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;

    // Zero out error block at the start
    if (VECTORIZED) {
        const int total_vecs = (B * static_cast<int>(R)) / 4;
        float4* err_vec_ptr = reinterpret_cast<float4*>(error_block);
        for (int idx = tid; idx < total_vecs; idx += blockDim.x) {
            err_vec_ptr[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    } else {
        const int total_err = B * static_cast<int>(R);
        for (int idx = tid; idx < total_err; idx += blockDim.x) {
            error_block[idx] = 0.0f;
        }
    }
    __syncthreads();

    // Serial Loop over Rows j
    for (int j = static_cast<int>(block_start); j < static_cast<int>(block_end); ++j) {
        int row_in_block = j - static_cast<int>(block_start);

        // Pointers for row j
        scalar_t* W_row = W + j * R;
        uint32_t* Q_row_vec = reinterpret_cast<uint32_t*>(qweight + j * R);
        float4* E_row_vec   = reinterpret_cast<float4*>(error_block + row_in_block * R);
        
        int R_vec           = static_cast<int>(R) / 4;
        int group_size_vec  = static_cast<int>(group_size) / 4;

        if (VECTORIZED) {
            for (int col_v = tid; col_v < R_vec; col_v += blockDim.x) {
                int g = col_v / group_size_vec;
                
                // Load QMeta directly (ALU bound, rely on L1 cache)
                const QMetaPacked m = qmeta[j * static_cast<int>(G) + g];
                float inv_s, s, q0, maxq_g;
                decode_qmeta(qmeta_to_u32(m), s, inv_s, q0, maxq_g, bits);
                int maxq_i = static_cast<int>(maxq_g);

                // Load 4 elements
                float x_vals[4];
                load_4(W_row + col_v * 4, x_vals);
                
                float e_vals[4], deq_vals[4];
                uint32_t q_packed;

                quantize_process_4(x_vals, inv_s, s, q0, maxq_i, e_vals, q_packed, deq_vals);

                // Store results
                store_4(W_row + col_v * 4, deq_vals);
                Q_row_vec[col_v] = q_packed;
                E_row_vec[col_v] = make_float4(e_vals[0], e_vals[1], e_vals[2], e_vals[3]);
            }
        } else {
            for (int col = tid; col < static_cast<int>(R); col += blockDim.x) {
                int g = col / static_cast<int>(group_size);
                
                const QMetaPacked m = qmeta[j * static_cast<int>(G) + g];
                float inv_s, s, q0, maxq_g;
                decode_qmeta(qmeta_to_u32(m), s, inv_s, q0, maxq_g, bits);
                int maxq_i = static_cast<int>(maxq_g);

                int idx_w = j * static_cast<int>(R) + col;
                float x = static_cast<float>(W[idx_w]);

                float e, deq;
                uint8_t q_byte;
                quantize_scalar(x, inv_s, s, q0, maxq_i, e, q_byte, deq);

                W[idx_w] = static_cast<scalar_t>(deq);
                qweight[idx_w] = q_byte;
                error_block[row_in_block * static_cast<int>(R) + col] = e;
            }
        }
        __syncthreads();

        // --- Update Step with Warp-Broadcast Alpha ---
        int block_rows_after_j = static_cast<int>(block_end) - (j + 1);
        if (block_rows_after_j > 0) {
            for (int rel_k = warp_id; rel_k < block_rows_after_j; rel_k += WARPS_PER_BLOCK) {
                int k = j + 1 + rel_k;
                int row_k_in_block = k - static_cast<int>(block_start);
                
                if (row_k_in_block >= B) continue;

                // Lane 0 loads alpha and broadcasts to warp
                float alpha = 0.0f;
                if (lane_id == 0) {
                    alpha = static_cast<float>(Hinv[j * static_cast<int>(C) + k]);
                }
                alpha = warp_broadcast(alpha, 0);

                if (VECTORIZED) {
                    scalar_t* W_k_row = W + k * R;
                    const float4* E_vec = reinterpret_cast<const float4*>(error_block + row_in_block * R);
                    int R_vec = static_cast<int>(R) / 4;

                    for (int col_v = lane_id; col_v < R_vec; col_v += 32) {
                        float w_vals[4];
                        load_4(W_k_row + col_v * 4, w_vals);
                        
                        float4 e_val = E_vec[col_v];
                        
                        w_vals[0] += alpha * e_val.x;
                        w_vals[1] += alpha * e_val.y;
                        w_vals[2] += alpha * e_val.z;
                        w_vals[3] += alpha * e_val.w;
                        
                        store_4(W_k_row + col_v * 4, w_vals);
                    }
                } else {
                    for (int col = lane_id; col < static_cast<int>(R); col += 32) {
                        int idx_e = row_in_block * static_cast<int>(R) + col;
                        int idx_w = k * static_cast<int>(R) + col;
                        float val = static_cast<float>(W[idx_w]);
                        val += alpha * error_block[idx_e];
                        W[idx_w] = static_cast<scalar_t>(val);
                    }
                }
            }
        }
        __syncthreads();
    }
}

} // anonymous namespace

// ----------------------------------------------------------------------------
// Host Wrapper
// ----------------------------------------------------------------------------

torch::Tensor gptq_solver_cuda(
    torch::Tensor weight,
    torch::Tensor hessian_inv,
    torch::Tensor qmeta_bytes,
    int64_t group_size,
    int64_t bits,
    int64_t block_size   // 0 => infer from SMEM limits
) {
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(hessian_inv.is_cuda(), "hessian_inv must be CUDA");
    TORCH_CHECK(qmeta_bytes.is_cuda(), "qmeta_bytes must be CUDA");
    
    weight      = weight.contiguous();
    hessian_inv = hessian_inv.contiguous();
    qmeta_bytes = qmeta_bytes.contiguous();

    const int64_t C = weight.size(0);
    const int64_t R = weight.size(1);

    TORCH_CHECK(hessian_inv.size(0) == C && hessian_inv.size(1) == C, "Hinv shape mismatch");

    int64_t G;
    if (qmeta_bytes.dim() == 3) {
        G = qmeta_bytes.size(1);
    } else if (qmeta_bytes.dim() == 2) {
        G = qmeta_bytes.size(0) / C;
    } else {
        TORCH_CHECK(false, "qmeta_bytes must be [C, G, 4] or [C*G, 4]");
    }

    TORCH_CHECK(G > 0, "Number of groups G must be > 0");

    if (block_size <= 0 || block_size > C) {
        block_size = 128; // Standard default without SMEM constraints
    }
    block_size = std::min<int64_t>(block_size, C);

    // 2. Output Allocation
    auto qweight = torch::empty({C, R}, torch::TensorOptions().dtype(torch::kUInt8).device(weight.device()));
    auto error_block = torch::empty({block_size, R}, torch::TensorOptions().dtype(at::kFloat).device(weight.device()));

    auto qmeta_flat = (qmeta_bytes.dim() == 3) ? qmeta_bytes.view({C * G, 4}) : qmeta_bytes;

    constexpr int WARPS_PER_BLOCK = 8;
    const int threads = WARPS_PER_BLOCK * 32;
    auto stream = at::cuda::getCurrentCUDAStream();

    const bool use_vectorized = (R % 4 == 0) && (group_size % 4 == 0);

    // Define dispatch macro
    #define LAUNCH_KERNEL(SCALAR_T) \
        if (use_vectorized) { \
            gptq_solver_kernel_no_smem<SCALAR_T, WARPS_PER_BLOCK, true><<< \
                1, threads, 0, stream>>>( \
                    weight.data_ptr<SCALAR_T>(), \
                    qweight.data_ptr<uint8_t>(), \
                    hessian_inv.data_ptr<SCALAR_T>(), \
                    reinterpret_cast<const QMetaPacked*>(qmeta_flat.data_ptr<uint8_t>()), \
                    error_block.data_ptr<float>(), \
                    C, R, G, \
                    block_start, block_end, \
                    group_size, \
                    static_cast<uint8_t>(bits) \
            ); \
        } else { \
            gptq_solver_kernel_no_smem<SCALAR_T, WARPS_PER_BLOCK, false><<< \
                1, threads, 0, stream>>>( \
                    weight.data_ptr<SCALAR_T>(), \
                    qweight.data_ptr<uint8_t>(), \
                    hessian_inv.data_ptr<SCALAR_T>(), \
                    reinterpret_cast<const QMetaPacked*>(qmeta_flat.data_ptr<uint8_t>()), \
                    error_block.data_ptr<float>(), \
                    C, R, G, \
                    block_start, block_end, \
                    group_size, \
                    static_cast<uint8_t>(bits) \
            ); \
        }

    for (int64_t block_start = 0; block_start < C; block_start += block_size) {
        int64_t block_end = std::min(block_start + block_size, C);
        int64_t B = block_end - block_start;

        // safety clear
        error_block.narrow(0, 0, B).zero_();

        // Dispatch based on weight type
        AT_DISPATCH_SWITCH(weight.scalar_type(), "gptq_solver_cuda",
            AT_DISPATCH_CASE(at::ScalarType::Float,   [&] { LAUNCH_KERNEL(float); })
            AT_DISPATCH_CASE(at::ScalarType::Half,    [&] { LAUNCH_KERNEL(at::Half); })
            AT_DISPATCH_CASE(at::ScalarType::BFloat16,[&] { LAUNCH_KERNEL(at::BFloat16); })
            AT_DISPATCH_CASE(at::ScalarType::Float8_e4m3fn, [&] { LAUNCH_KERNEL(c10::Float8_e4m3fn); })
        );

        CUDA_CHECK(cudaGetLastError());

        // 5. Tail Update (cuBLAS via Torch)
        if (block_end < C) {
            auto W_tail  = weight.narrow(0, block_end, C - block_end);
            auto H_block = hessian_inv.narrow(0, block_start, B).narrow(1, block_end, C - block_end);
            auto err_sub = error_block.narrow(0, 0, B);

            if (weight.scalar_type() == at::ScalarType::Float8_e4m3fn) {
                // FP8 Workaround: Upcast to FP16 (Half) for matrix multiplication
                auto W_tail_h = W_tail.to(torch::kHalf);
                auto H_block_h = H_block.to(torch::kHalf);
                auto err_sub_h = err_sub.to(torch::kHalf);
                
                W_tail_h.addmm_(H_block_h.transpose(0, 1), err_sub_h, 1.0f, 1.0f);
                W_tail.copy_(W_tail_h);
            } else if (weight.scalar_type() != at::ScalarType::Float) {
                // Half/BFloat16: Cast error (float) to match weight for addmm
                auto err_sub_casted = err_sub.to(weight.scalar_type());
                W_tail.addmm_(H_block.transpose(0, 1), err_sub_casted, 1.0f, 1.0f);
            } else {
                // Float32
                W_tail.addmm_(H_block.transpose(0, 1), err_sub, 1.0f, 1.0f);
            }
        }
    }

    return qweight;
}