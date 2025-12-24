// solver_babai.cu
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>
#include <algorithm>

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

struct QMetaPacked {
    int16_t  log2_scale_fp;  // Q8.8 fixed-point log2(scale)
    uint8_t  qzero;
    uint8_t  flags;          // bit0 = symmetric
};
static_assert(sizeof(QMetaPacked) == 4, "QMetaPacked must be 4 bytes.");

__device__ __forceinline__ uint32_t qmeta_to_u32(const QMetaPacked& m) {
    return *reinterpret_cast<const uint32_t*>(&m);
}

__device__ __forceinline__ void decode_qmeta(
    uint32_t packed,
    float&   scale,
    float&   inv_scale,
    float&   qzero_f,
    uint8_t  bits
) {
    int16_t log2_q88 = static_cast<int16_t>(packed & 0xFFFFu);
    uint8_t qzero_u8 = static_cast<uint8_t>((packed >> 16) & 0xFFu);
    uint8_t flags    = static_cast<uint8_t>(packed >> 24);

    constexpr float INV256 = 1.0f / 256.0f;
    float log2_scale = __int2float_rn(log2_q88) * INV256;
    scale     = exp2f(log2_scale);
    inv_scale = exp2f(-log2_scale);

    if (flags & 0x01) {
        int maxq_i = (1 << bits) - 1;
        qzero_u8 = static_cast<uint8_t>((static_cast<float>(maxq_i) + 1.0f) * 0.5f);
    }
    qzero_f = static_cast<float>(qzero_u8);
}

__device__ __forceinline__ void quantize_scalar(
    float x, float inv_s, float s, float q0, int maxq_i,
    float& err_out, uint8_t& q_out, float& deq_out
) {
    float biased = x * inv_s + q0;
    int q = __float2int_rn(biased);

    q = (q < 0) ? 0 : q;
    q = (q > maxq_i) ? maxq_i : q;

    deq_out = __fmaf_rn(static_cast<float>(q), s, -q0 * s);
    err_out = x - deq_out;
    q_out   = static_cast<uint8_t>(q);
}

// --- dtype helpers ----------------------------------------------------------

template <typename T>
__device__ __forceinline__ float to_f32(T x) { return static_cast<float>(x); }

template <>
__device__ __forceinline__ float to_f32<c10::Float8_e4m3fn>(c10::Float8_e4m3fn x) {
    return static_cast<float>(x);
}

template <typename T>
__device__ __forceinline__ T from_f32(float x) { return static_cast<T>(x); }

template <>
__device__ __forceinline__ c10::Float8_e4m3fn from_f32<c10::Float8_e4m3fn>(float x) {
    return static_cast<c10::Float8_e4m3fn>(x);
}

// -----------------------------------------------------------------------------
// Scratch fill kernels for addmm_ path
// -----------------------------------------------------------------------------

template <typename scalar_t>
__global__ void fill_A_scaled_kernel(
    scalar_t* __restrict__ A_tmp,     // [C, block_size] row-major
    const float* __restrict__ A,      // [C, C] float
    const float* __restrict__ invD,   // [C] float
    int C, int block_size,
    int block_start, int B
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x; // 0..B-1
    int i = blockIdx.y * blockDim.y + threadIdx.y; // 0..block_start-1
    if (k >= B || i >= block_start) return;

    float a = A[i * C + (block_start + k)];
    float s = a * invD[i];
    A_tmp[i * block_size + k] = from_f32<scalar_t>(s);
}

template <typename scalar_t>
__global__ void cast_E_kernel(
    scalar_t* __restrict__ E_out,     // [block_size, R]
    const float* __restrict__ E_in,   // [B, R]
    int R, int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * R;
    if (idx >= total) return;
    int k = idx / R;
    int r = idx - k * R;
    E_out[k * R + r] = from_f32<scalar_t>(E_in[k * R + r]);
}

// -----------------------------------------------------------------------------
// Kernel 1: Babai quantize block (back-to-front) + intra-block propagation.
//
// - Builds S_sh(i,t) = (A(i,t)/A(i,i)) = A(i,t) * invD(i) in shared.
// - Triangle hygiene: only load needed upper entries (t > i); others set to 0.
// - Warp decode optimization uses ACTIVE mask and broadcasts from an active lane.
// -----------------------------------------------------------------------------

template <typename scalar_t, int MAX_B>
__global__ void babai_quant_block_kernel_fast(
    scalar_t* __restrict__ W,                   // [C, R]
    uint8_t*  __restrict__ qweight,             // [C, R]
    const QMetaPacked* __restrict__ qmeta,      // [R * G]
    const float* __restrict__ A,                // [C, C] float (upper-tri expected)
    const float* __restrict__ invD_all,         // [C] float
    float* __restrict__ Eblk,                   // [B, R] float
    int C, int R, int G,
    int block_start, int B,
    int group_size,
    uint8_t bits
) {
    __shared__ float S_sh[MAX_B * MAX_B];

    // Build row-scaled block in shared: S(i,t) = A(i,t)*invD(i), only for t>i
    for (int idx = threadIdx.x; idx < B * B; idx += blockDim.x) {
        int i = idx / B;
        int t = idx - i * B;

        float v = 0.0f;
        if (t > i) {
            float invd = invD_all[block_start + i];
            float a    = A[(block_start + i) * C + (block_start + t)];
            v = a * invd;
        }
        S_sh[i * MAX_B + t] = v;
    }
    __syncthreads();

    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;

    // --- Correctness: derive active mask AFTER sync; do not use full mask on edge tile.
    const unsigned full = 0xFFFFFFFFu;
    //if (mask == 0) return; // whole warp out-of-range (uniform)

    int lane = threadIdx.x & 31;
    //if ((mask & (1u << lane)) == 0) return; // inactive lane exits safely

    // pick an active source lane for broadcasts

    // Load this column across the block into registers
    float x[MAX_B];
    

    for (int i = 0; i < MAX_B; ++i) x[i] = 0.0f;

    for (int i = 0; i < B; ++i) {
        int row = block_start + i;
        x[i] = to_f32(W[row * R + r]);
    }

    const int maxq_i = (1 << bits) - 1;

    // Back-to-front
    for (int t = B - 1; t >= 0; --t) {
        int row = block_start + t;

        int g = row / group_size;
        if (g >= G) g = G - 1;   // or return; but clamping matches your “tail group” idea

        float s = 0.f, inv_s = 0.f, q0 = 0.f;
        
        uint32_t packed = qmeta_to_u32(qmeta[r * G + g]);
        decode_qmeta(packed, s, inv_s, q0, bits);
        

        float err, deq;
        uint8_t qb;
        quantize_scalar(x[t], inv_s, s, q0, maxq_i, err, qb, deq);

        qweight[row * R + r] = qb;
        W[row * R + r]       = deq;
        Eblk[t * R + r]      = err;

        // Propagate to earlier rows i < t
        #pragma unroll
        for (int i = 0; i < MAX_B; ++i) {
            if (i < t) {
                float alpha = S_sh[i * MAX_B + t]; // already 0 for t<=i
                x[i] = __fmaf_rn(alpha, err, x[i]);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel 2: Prefix update for FP8 (custom), float accumulate.
// W[i,r] += sum_k (A[i, block_start+k] * invD[i]) * Eblk[k,r]
// -----------------------------------------------------------------------------

template <int MAX_B, int TILE_R, int TILE_I>
__global__ void babai_update_left_fp8_kernel(
    c10::Float8_e4m3fn* __restrict__ W,     // [C, R]
    const float* __restrict__ A,            // [C, C]
    const float* __restrict__ invD_all,     // [C]
    const float* __restrict__ Eblk,         // [B, R]
    int C, int R,
    int block_start, int B
) {
    __shared__ float E_sh[MAX_B * TILE_R];   // [B][TILE_R]
    __shared__ float S_sh[TILE_I * MAX_B];   // [TILE_I][B]

    int r0 = blockIdx.x * TILE_R;
    int i0 = blockIdx.y * TILE_I;

    int tx = threadIdx.x; // 0..TILE_R-1
    int ty = threadIdx.y; // 0..TILE_I-1

    // load E tile
    for (int k = ty; k < B; k += TILE_I) {
        int r = r0 + tx;
        E_sh[k * TILE_R + tx] = (r < R) ? Eblk[k * R + r] : 0.f;
    }

    int i = i0 + ty;
    if (i < block_start) {
        float invd = invD_all[i];
        for (int k = tx; k < B; k += TILE_R) {
            float a = A[i * C + (block_start + k)];
            S_sh[ty * MAX_B + k] = a * invd;
        }
    } else {
        for (int k = tx; k < B; k += TILE_R) {
            S_sh[ty * MAX_B + k] = 0.f;
        }
    }

    __syncthreads();

    int r = r0 + tx;
    if (i < block_start && r < R) {
        float acc = 0.f;
        #pragma unroll
        for (int k = 0; k < MAX_B; ++k) {
            if (k < B) {
                acc = __fmaf_rn(S_sh[ty * MAX_B + k], E_sh[k * TILE_R + tx], acc);
            }
        }
        float w = static_cast<float>(W[i * R + r]);
        w += acc;
        W[i * R + r] = static_cast<c10::Float8_e4m3fn>(w);
    }
}

} // namespace

// -----------------------------------------------------------------------------
// Host wrapper
// -----------------------------------------------------------------------------
torch::Tensor babai_solver_cuda(
    torch::Tensor weight,      // [C, R]
    torch::Tensor A,           // [C, C] upper-tri = chol(H)^T
    torch::Tensor qmeta_bytes, // [R, G, 4] or [R*G, 4]
    int64_t group_size,
    int64_t bits,
    int64_t block_size
) {
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(A.is_cuda(),      "A must be CUDA");
    TORCH_CHECK(qmeta_bytes.is_cuda(), "qmeta_bytes must be CUDA");


    qmeta_bytes = qmeta_bytes.contiguous();

    const int64_t C = weight.size(0);
    const int64_t R = weight.size(1);

    TORCH_CHECK(weight.dim() == 2, "weight must be [C, R]");
    TORCH_CHECK(A.dim() == 2 && A.size(0) == C && A.size(1) == C, "A must be [C, C]");

    // Determine G
    
    int64_t G;
    if (qmeta_bytes.dim() == 3) {
        TORCH_CHECK(qmeta_bytes.size(0) == R, "qmeta_bytes[0] must be R");
        TORCH_CHECK(qmeta_bytes.size(2) == 4, "qmeta_bytes[...,4] expected");
        G = qmeta_bytes.size(1);
    } else {
        TORCH_CHECK(qmeta_bytes.dim() == 2 && qmeta_bytes.size(1) == 4,
                    "qmeta_bytes must be [R,G,4] or [R*G,4]");
        TORCH_CHECK(qmeta_bytes.size(0) % R == 0, "qmeta_bytes[0] must be multiple of R");
        G = qmeta_bytes.size(0) / R;
    }
    

    constexpr int MAX_B = 32;
    if (block_size <= 0 || block_size > MAX_B) block_size = MAX_B;
    if (block_size > C) block_size = C;

    auto qweight = torch::zeros({C, R},
        torch::TensorOptions().dtype(torch::kUInt8).device(weight.device()));

    // Eblk (float) reusable
    auto Eblk = torch::zeros({block_size, R},
        torch::TensorOptions().dtype(at::kFloat).device(weight.device()));

    auto qmeta_flat =
        (qmeta_bytes.dim() == 3) ? qmeta_bytes.view({R * G, 4}) : qmeta_bytes;

    // Cast A to float once for reads
    auto A_f = (A.scalar_type() == at::ScalarType::Float) ? A : A.to(torch::kFloat);
    A_f = A_f.contiguous();
    weight = (weight.scalar_type() == at::ScalarType::Float) ? weight : weight.to(torch::kFloat);
    weight = weight.contiguous();


    // invD_all[i] = 1 / A[i,i]
    auto invD_all = A_f.diagonal(0, 0, 1).reciprocal().contiguous();

    auto stream = at::cuda::getCurrentCUDAStream();
    auto st = weight.scalar_type();

    // Scratch for addmm_ path (reused, no per-iter allocs)
    torch::Tensor A_tmp, E_tmp;
    if (st == at::ScalarType::Half || st == at::ScalarType::BFloat16 || st == at::ScalarType::Float) {
        A_tmp = torch::zeros({C, block_size}, torch::TensorOptions().dtype(st).device(weight.device()));
        if (st != at::ScalarType::Float) {
            E_tmp = torch::zeros({block_size, R}, torch::TensorOptions().dtype(st).device(weight.device()));
        }
    }

    constexpr int THREADS_Q = 128;

    // fp8 prefix update tiling
    constexpr int TILE_R = 64;
    constexpr int TILE_I = 4;
    dim3 fp8_block(TILE_R, TILE_I);

    // Right-to-left blocks
    for (int64_t block_end = C; block_end > 0; block_end -= block_size) {
        const int64_t block_start = std::max<int64_t>(0, block_end - block_size);
        const int64_t B_long      = block_end - block_start;
        const int     B           = static_cast<int>(B_long);

        auto Eblk_view = Eblk.narrow(0, 0, B_long); // [B, R]

        // 1) Quantize block + intra-block propagation
        const int grid_q = (static_cast<int>(R) + THREADS_Q - 1) / THREADS_Q;

        babai_quant_block_kernel_fast<float, MAX_B><<<grid_q, THREADS_Q, 0, stream>>>(
                weight.data_ptr<float>(),
                qweight.data_ptr<uint8_t>(),
                reinterpret_cast<const QMetaPacked*>(qmeta_flat.data_ptr<uint8_t>()),
                A_f.data_ptr<float>(),
                invD_all.data_ptr<float>(),
                Eblk_view.data_ptr<float>(),
                (int)C, (int)R, (int)G,
                (int)block_start, B,
                (int)group_size,
                (uint8_t)bits
        );
        

        // 2) Prefix update (rows [0, block_start))
        if (block_start > 0) {
                // Fill A_tmp slice and E_tmp (if needed), then cuBLAS addmm_
                const int tx = 16, ty = 16;
                dim3 blk(tx, ty);
                dim3 grd((B + tx - 1) / tx, ((int)block_start + ty - 1) / ty);

                fill_A_scaled_kernel<float><<<grd, blk, 0, stream>>>(
                        A_tmp.data_ptr<float>(),
                        A_f.data_ptr<float>(),
                        invD_all.data_ptr<float>(),
                        (int)C, (int)block_size,
                        (int)block_start, B
                );

                auto W_left = weight.narrow(0, 0, block_start);
                auto A_view = A_tmp.narrow(0, 0, block_start).narrow(1, 0, B_long);
                W_left.addmm_(A_view, Eblk_view, /*beta=*/1.0, /*alpha=*/1.0);
                
        }
    }
    

    return qweight;
}