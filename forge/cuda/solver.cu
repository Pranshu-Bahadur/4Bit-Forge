#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>
#include <tuple>

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

__device__ __forceinline__ float warp_broadcast(float v, int src_lane = 0) {
    return __shfl_sync(0xffffffff, v, src_lane);
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
    // Layout: [15:0] log2_scale_fp (Q8.8), [23:16] qzero, [31:24] flags
    int16_t log2_q88 = static_cast<int16_t>(packed & 0xFFFFu);
    uint8_t qzero_u8 = static_cast<uint8_t>((packed >> 16) & 0xFFu);
    uint8_t flags    = static_cast<uint8_t>(packed >> 24);

    // log2(scale) = log2_q88 / 256
    constexpr float INV256 = 1.0f / 256.0f;
    float log2_scale = __int2float_rn(log2_q88) * INV256;
    scale     = exp2f(log2_scale);
    inv_scale = exp2f(-log2_scale); 

    // Bits-per-group
    uint8_t bits_g = global_bits;
    int maxq_i = (1 << bits_g) - 1;
    maxq_g = static_cast<float>(maxq_i);

    // Symmetric override: qzero = (maxq + 1)/2
    if (flags & 0x01) {
        constexpr float HALF = 0.5f;
        qzero_u8 = static_cast<uint8_t>((maxq_g + 1.0f) * HALF);
    }
    qzero_f = static_cast<float>(qzero_u8);
}

// --- Quantization Primitives ------------------------------------------------

// Scalar quantization logic
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

// Vectorized (Vec4) quantization logic
__device__ __forceinline__ void quantize_vec4(
    float4 x_vec, float inv_s, float s, float q0, int maxq_i,
    float4& e_vec, uint32_t& q_packed, float4& deq_vec
) {
    uint8_t q0_b, q1_b, q2_b, q3_b;
    
    quantize_scalar(x_vec.x, inv_s, s, q0, maxq_i, e_vec.x, q0_b, deq_vec.x);
    quantize_scalar(x_vec.y, inv_s, s, q0, maxq_i, e_vec.y, q1_b, deq_vec.y);
    quantize_scalar(x_vec.z, inv_s, s, q0, maxq_i, e_vec.z, q2_b, deq_vec.z);
    quantize_scalar(x_vec.w, inv_s, s, q0, maxq_i, e_vec.w, q3_b, deq_vec.w);

    // Pack 4 bytes into 1 uint32 (Little Endian: x=LSB, w=MSB)
    q_packed = (uint32_t)q0_b | 
               ((uint32_t)q1_b << 8) | 
               ((uint32_t)q2_b << 16) | 
               ((uint32_t)q3_b << 24);
}

// ----------------------------------------------------------------------------
// Main Kernel
// ----------------------------------------------------------------------------
// Templated for VECTORIZED path (R % 4 == 0) vs SCALAR path (fallback)
//
template <int WARPS_PER_BLOCK, bool VECTORIZED>
__global__ void gptq_solver_kernel(
    float* __restrict__ W,            // (C, R)
    uint8_t* __restrict__ qweight,    // (C, R)
    const float* __restrict__ Hinv,   // (C, C)
    const QMetaPacked* __restrict__ qmeta, 
    float* __restrict__ error_block,  // (B, R)
    int64_t C, int64_t R, int64_t G,
    int64_t block_start, int64_t block_end,
    int64_t group_size,
    uint8_t bits
) {
    const int B = static_cast<int>(block_end - block_start);
    if (B <= 0) return;

    // Shared Memory Setup (Padded for bank conflicts)
    const int STRIDE = ((int)G + 7) & ~7;
    extern __shared__ float smem[];
    float* sm_inv_scale = smem;
    float* sm_scale     = sm_inv_scale + B * STRIDE;
    float* sm_qzero_f   = sm_scale     + B * STRIDE;
    float* sm_maxq_g    = sm_qzero_f   + B * STRIDE;

    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;

    // 1. Cooperative Load: qmeta -> smem
    const int total_qmeta = B * static_cast<int>(G);
    for (int idx = tid; idx < total_qmeta; idx += blockDim.x) {
        int row_in_block = idx / static_cast<int>(G);
        int g            = idx % static_cast<int>(G);
        int j            = static_cast<int>(block_start) + row_in_block;

        const QMetaPacked m = qmeta[j * static_cast<int>(G) + g];
        
        float scale, inv_scale, qzero_f, maxq_g;
        decode_qmeta(qmeta_to_u32(m), scale, inv_scale, qzero_f, maxq_g, bits);

        int off = row_in_block * STRIDE + g;
        sm_inv_scale[off] = inv_scale;
        sm_scale[off]     = scale;
        sm_qzero_f[off]   = qzero_f;
        sm_maxq_g[off]    = maxq_g;
    }
    __syncthreads();

    // 2. Zero Error Block
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

    // 3. Serial Loop over Rows j
    for (int j = static_cast<int>(block_start); j < static_cast<int>(block_end); ++j) {
        int row_in_block = j - static_cast<int>(block_start);

        // --- 3a. Quantize Row j ---
        if (VECTORIZED) {
            float4* W_row_vec   = reinterpret_cast<float4*>(W + j * R);
            uint32_t* Q_row_vec = reinterpret_cast<uint32_t*>(qweight + j * R);
            float4* E_row_vec   = reinterpret_cast<float4*>(error_block + row_in_block * R);
            int R_vec           = static_cast<int>(R) / 4;
            int group_size_vec  = static_cast<int>(group_size) / 4;

            for (int col_v = tid; col_v < R_vec; col_v += blockDim.x) {
                int g = col_v / group_size_vec;
                
                int off = row_in_block * STRIDE + g;
                float inv_s = sm_inv_scale[off];
                float s     = sm_scale[off];
                float q0    = sm_qzero_f[off];
                int maxq_i  = static_cast<int>(sm_maxq_g[off]);

                float4 x_vec = W_row_vec[col_v];
                
                float4 e_vec, deq_vec;
                uint32_t q_packed;

                quantize_vec4(x_vec, inv_s, s, q0, maxq_i, e_vec, q_packed, deq_vec);

                W_row_vec[col_v] = deq_vec;
                Q_row_vec[col_v] = q_packed;
                E_row_vec[col_v] = e_vec;
            }
        } else {
            for (int col = tid; col < static_cast<int>(R); col += blockDim.x) {
                int g = col / static_cast<int>(group_size);
                int off = row_in_block * STRIDE + g;
                
                float inv_s = sm_inv_scale[off];
                float s     = sm_scale[off];
                float q0    = sm_qzero_f[off];
                int maxq_i  = static_cast<int>(sm_maxq_g[off]);

                int idx_w = j * static_cast<int>(R) + col;
                float x = W[idx_w];

                float e, deq;
                uint8_t q_byte;
                quantize_scalar(x, inv_s, s, q0, maxq_i, e, q_byte, deq);

                W[idx_w] = deq;
                qweight[idx_w] = q_byte;
                error_block[row_in_block * static_cast<int>(R) + col] = e;
            }
        }
        __syncthreads();

        // --- 3b. Update Step ---
        int block_rows_after_j = static_cast<int>(block_end) - (j + 1);
        if (block_rows_after_j > 0) {
            for (int rel_k = warp_id; rel_k < block_rows_after_j; rel_k += WARPS_PER_BLOCK) {
                int k = j + 1 + rel_k;
                int row_k_in_block = k - static_cast<int>(block_start);
                
                if (row_k_in_block >= B) continue;

                float alpha = 0.0f;
                if (lane_id == 0) alpha = Hinv[j * static_cast<int>(C) + k];
                alpha = warp_broadcast(alpha, 0);

                if (VECTORIZED) {
                    float4* W_k_vec = reinterpret_cast<float4*>(W + k * R);
                    const float4* E_vec = reinterpret_cast<const float4*>(error_block + row_in_block * R);
                    int R_vec = static_cast<int>(R) / 4;

                    for (int col_v = lane_id; col_v < R_vec; col_v += 32) {
                        float4 w_val = W_k_vec[col_v];
                        float4 e_val = E_vec[col_v];
                        
                        w_val.x += alpha * e_val.x;
                        w_val.y += alpha * e_val.y;
                        w_val.z += alpha * e_val.z;
                        w_val.w += alpha * e_val.w;
                        
                        W_k_vec[col_v] = w_val;
                    }
                } else {
                    for (int col = lane_id; col < static_cast<int>(R); col += 32) {
                        int idx_e = row_in_block * static_cast<int>(R) + col;
                        int idx_w = k * static_cast<int>(R) + col;
                        W[idx_w] += alpha * error_block[idx_e];
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
        // (C, G, 4)
        TORCH_CHECK(qmeta_bytes.size(0) == C, "qmeta_bytes[0] must be C");
        TORCH_CHECK(qmeta_bytes.size(2) == 4, "qmeta_bytes last dim must be 4");
        G = qmeta_bytes.size(1);
    } else if (qmeta_bytes.dim() == 2) {
        // (C*G, 4)
        TORCH_CHECK(qmeta_bytes.size(1) == 4, "qmeta_bytes last dim must be 4");
        TORCH_CHECK(qmeta_bytes.size(0) % C == 0, "qmeta_bytes[0] must be multiple of C");
        G = qmeta_bytes.size(0) / C;
    } else {
        TORCH_CHECK(false, "qmeta_bytes must be [C, G, 4] or [C*G, 4]");
    }

    TORCH_CHECK(G > 0, "Number of groups G must be > 0");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "Solver requires float32 weight");
    TORCH_CHECK(hessian_inv.scalar_type() == at::kFloat, "Solver requires float32 Hinv");

    // Shared memory–bounded block_size inference
    const int STRIDE = ((int)G + 7) & ~7;
    const size_t bytes_per_row = 4ull * static_cast<size_t>(STRIDE) * sizeof(float); // 4 arrays

    const auto* prop = at::cuda::getCurrentDeviceProperties();
    size_t max_smem = static_cast<size_t>(prop->sharedMemPerBlock);  // conservative: default dyn. smem

    size_t max_B_from_smem = max_smem / (bytes_per_row > 0 ? bytes_per_row : 1);
    if (max_B_from_smem == 0) {
        max_B_from_smem = 1;
    }

    int64_t user_block = block_size;
    if (user_block <= 0 || user_block > C) {
        user_block = C;
    }
    int64_t smem_block = static_cast<int64_t>(max_B_from_smem);
    if (smem_block <= 0) smem_block = 1;

    block_size = std::min<int64_t>(user_block, smem_block);
    block_size = std::max<int64_t>(block_size, 1);

    // 2. Output Allocation
    auto qweight = torch::empty({C, R}, torch::TensorOptions().dtype(torch::kUInt8).device(weight.device()));
    auto error_block = torch::empty({block_size, R}, torch::TensorOptions().dtype(at::kFloat).device(weight.device()));

    auto qmeta_flat = (qmeta_bytes.dim() == 3)
        ? qmeta_bytes.view({C * G, 4})
        : qmeta_bytes;

    constexpr int WARPS_PER_BLOCK = 8;
    const int threads = WARPS_PER_BLOCK * 32;
    auto stream = at::cuda::getCurrentCUDAStream();

    const bool use_vectorized = (R % 4 == 0) && (group_size % 4 == 0);

    for (int64_t block_start = 0; block_start < C; block_start += block_size) {
        int64_t block_end = std::min(block_start + block_size, C);
        int64_t B = block_end - block_start;

        int STRIDE_local = ((int)G + 7) & ~7;
        size_t smem_bytes = 4ull * static_cast<size_t>(B) * static_cast<size_t>(STRIDE_local) * sizeof(float);

        // safety clear
        error_block.narrow(0, 0, B).zero_();

        if (use_vectorized) {
            gptq_solver_kernel<WARPS_PER_BLOCK, true><<<
                1, threads, smem_bytes, stream>>>(
                    weight.data_ptr<float>(),
                    qweight.data_ptr<uint8_t>(),
                    hessian_inv.data_ptr<float>(),
                    reinterpret_cast<const QMetaPacked*>(qmeta_flat.data_ptr<uint8_t>()),
                    error_block.data_ptr<float>(),
                    C, R, G,
                    block_start, block_end,
                    group_size,
                    static_cast<uint8_t>(bits)
            );
        } else {
            gptq_solver_kernel<WARPS_PER_BLOCK, false><<<
                1, threads, smem_bytes, stream>>>(
                    weight.data_ptr<float>(),
                    qweight.data_ptr<uint8_t>(),
                    hessian_inv.data_ptr<float>(),
                    reinterpret_cast<const QMetaPacked*>(qmeta_flat.data_ptr<uint8_t>()),
                    error_block.data_ptr<float>(),
                    C, R, G,
                    block_start, block_end,
                    group_size,
                    static_cast<uint8_t>(bits)
            );
        }

        CUDA_CHECK(cudaGetLastError());

        // 5. Tail Update (cuBLAS via Torch)
        if (block_end < C) {
            auto W_tail  = weight.narrow(0, block_end, C - block_end);
            auto H_block = hessian_inv.narrow(0, block_start, B).narrow(1, block_end, C - block_end);
            auto err_sub = error_block.narrow(0, 0, B);

            // W_tail += H_block.T @ err_sub
            W_tail.addmm_(H_block.transpose(0, 1), err_sub, 1.0f, 1.0f);
        }
    }

    return qweight;
}
