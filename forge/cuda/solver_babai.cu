// solver_babai.cu
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>
#include <algorithm>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t _err = (expr);                                                \
    TORCH_CHECK(_err == cudaSuccess, "CUDA error: ",                          \
                cudaGetErrorString(_err), " at ", __FILE__, ":", __LINE__);   \
  } while (0)

namespace {

struct QMetaPacked {
    int16_t  log2_scale_fp;  // Q8.8 fixed-point log2(scale)
    uint8_t  qzero;          // uint8
    uint8_t  flags;          // bit0 = symmetric
};
static_assert(sizeof(QMetaPacked) == 4, "QMetaPacked must be 4 bytes.");

__device__ __forceinline__ uint32_t qmeta_to_u32(const QMetaPacked* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}

// -----------------------------
// W-dtype ops (no explicit fp32 math in propagation/update)
// Only qmeta decode uses fp32, then downcasts.
// -----------------------------

template <typename T>
struct WOps;

// float
template <>
struct WOps<float> {
    using T = float;
    __device__ __forceinline__ static T zero() { return 0.0f; }
    __device__ __forceinline__ static T add(T a, T b) { return a + b; }
    __device__ __forceinline__ static T sub(T a, T b) { return a - b; }
    __device__ __forceinline__ static T mul(T a, T b) { return a * b; }
    __device__ __forceinline__ static T fma(T a, T b, T c) { return __fmaf_rn(a, b, c); }
    __device__ __forceinline__ static float to_f32(T a) { return a; }
    __device__ __forceinline__ static T from_f32(float x) { return x; }
};

// fp16
template <>
struct WOps<__half> {
    using T = __half;
    __device__ __forceinline__ static T zero() { return __float2half_rn(0.0f); }
    __device__ __forceinline__ static T add(T a, T b) { return __hadd(a, b); }
    __device__ __forceinline__ static T sub(T a, T b) { return __hsub(a, b); }
    __device__ __forceinline__ static T mul(T a, T b) { return __hmul(a, b); }
    __device__ __forceinline__ static T fma(T a, T b, T c) {
#if defined(__CUDA_ARCH__)
        return __hfma(a, b, c);
#else
        // host compile path (shouldn't execute)
        return __hadd(__hmul(a, b), c);
#endif
    }
    __device__ __forceinline__ static float to_f32(T a) { return __half2float(a); }
    __device__ __forceinline__ static T from_f32(float x) { return __float2half_rn(x); }
};

// bf16
template <>
struct WOps<__nv_bfloat16> {
    using T = __nv_bfloat16;
    __device__ __forceinline__ static T zero() { return __float2bfloat16_rn(0.0f); }
    __device__ __forceinline__ static T add(T a, T b) { return __hadd(a, b); }
    __device__ __forceinline__ static T sub(T a, T b) { return __hsub(a, b); }
    __device__ __forceinline__ static T mul(T a, T b) { return __hmul(a, b); }
    __device__ __forceinline__ static T fma(T a, T b, T c) {
        // Use mul+add to keep it strictly in bf16 ops.
        return add(mul(a, b), c);
    }
    __device__ __forceinline__ static float to_f32(T a) { return __bfloat162float(a); }
    __device__ __forceinline__ static T from_f32(float x) { return __float2bfloat16_rn(x); }
};

// -----------------------------
// qmeta decode: fp32 -> downcast to W dtype
// -----------------------------
template <typename scalar_t>
__device__ __forceinline__ void decode_qmeta_to_wdtype(
    const QMetaPacked* qmeta_ptr,
    uint8_t bits,
    scalar_t& s_t, scalar_t& invs_t, scalar_t& q0_t
) {
    uint32_t packed = qmeta_to_u32(qmeta_ptr);

    int16_t log2_q88 = static_cast<int16_t>(packed & 0xFFFFu);
    uint8_t qzero_u8 = static_cast<uint8_t>((packed >> 16) & 0xFFu);
    uint8_t flags    = static_cast<uint8_t>(packed >> 24);

    // fp32 decode
    constexpr float INV256 = 1.0f / 256.0f;
    float log2_scale = static_cast<float>(log2_q88) * INV256;
    float s  = exp2f(log2_scale);
    float is = exp2f(-log2_scale);

    float q0 = static_cast<float>(qzero_u8);
    if (flags & 0x01) {
        int maxq_i = (1 << bits) - 1;
        q0 = (static_cast<float>(maxq_i) + 1.0f) * 0.5f;
    }

    // downcast to W dtype
    s_t    = WOps<scalar_t>::from_f32(s);
    invs_t = WOps<scalar_t>::from_f32(is);
    q0_t   = WOps<scalar_t>::from_f32(q0);
}

// -----------------------------
// quantize step: computations in W dtype; only rounding uses fp32 convert of biased
// -----------------------------
template <typename scalar_t>
__device__ __forceinline__ void quantize_scalar_wdtype(
    scalar_t x,
    scalar_t inv_s,
    scalar_t s,
    scalar_t q0,
    int maxq,
    scalar_t& err_out,
    uint8_t& q_out,
    scalar_t& deq_out
) {
    // biased = x * inv_s + q0  (W dtype)
    scalar_t biased_t = WOps<scalar_t>::fma(x, inv_s, q0);

    // rounding decision needs float
    float biased_f = WOps<scalar_t>::to_f32(biased_t);
    int q = __float2int_rn(biased_f);
    q = (q < 0) ? 0 : q;
    q = (q > maxq) ? maxq : q;

    // deq = (q - q0) * s  (W dtype)
    scalar_t q_t   = WOps<scalar_t>::from_f32(static_cast<float>(q));
    scalar_t dq_t  = WOps<scalar_t>::sub(q_t, q0);
    scalar_t deq_t = WOps<scalar_t>::mul(dq_t, s);

    // err = x - deq (W dtype)
    scalar_t err_t = WOps<scalar_t>::sub(x, deq_t);

    q_out   = static_cast<uint8_t>(q);
    deq_out = deq_t;
    err_out = err_t;
}

// ============================================================================
// Kernel 1: Babai quantize block (back-to-front) + intra-block propagation.
// Everything stored/propagated in scalar_t (W dtype).
// ============================================================================
template <typename scalar_t, int MAX_B>
__global__ void babai_quant_block_kernel_wdtype(
    scalar_t* __restrict__ W,                   // [C, R]
    uint8_t*  __restrict__ qweight,             // [C, R]
    const QMetaPacked* __restrict__ qmeta,      // [C * G]
    const scalar_t* __restrict__ A,             // [C, C] upper-tri (W dtype)
    const scalar_t* __restrict__ invD_all,      // [C] invD = 1/Aii (W dtype)
    scalar_t* __restrict__ Eblk,                // [B, R] (W dtype)
    int C, int R, int G,
    int block_start, int B,
    int group_size,
    uint8_t bits
) {
    __shared__ scalar_t S_sh[MAX_B * MAX_B];

    // Build S_sh(i,t) = A(i,t) * invD(i) for t>i else 0, entirely in W dtype ops.
    for (int idx = threadIdx.x; idx < B * B; idx += blockDim.x) {
        int i = idx / B;
        int t = idx - i * B;

        scalar_t v = WOps<scalar_t>::zero();
        if (t > i) {
            int gi = block_start + i;
            int gt = block_start + t;
            scalar_t a   = A[gi * C + gt];
            scalar_t inv = invD_all[gi];
            v = WOps<scalar_t>::mul(a, inv);
        }
        S_sh[i * MAX_B + t] = v;
    }
    __syncthreads();

    int r = blockIdx.x * blockDim.x + threadIdx.x;

    // Active mask for edge R
    const unsigned full = 0xFFFFFFFFu;
    unsigned mask = __ballot_sync(full, r < R);
    if (mask == 0) return;
    int lane = threadIdx.x & 31;
    if ((mask & (1u << lane)) == 0) return;

    int src = __ffs(mask) - 1;

    // Load this column across the block into registers (W dtype)
    scalar_t x[MAX_B];
#pragma unroll
    for (int i = 0; i < MAX_B; ++i) x[i] = WOps<scalar_t>::zero();

    for (int i = 0; i < B; ++i) {
        int row = block_start + i;
        x[i] = W[row * R + r];
    }

    const int maxq_i = (1 << bits) - 1;

    // Back-to-front within block
    for (int t = B - 1; t >= 0; --t) {
        int row = block_start + t;

        // group id per lane
        int g  = r / group_size;
        int g0 = __shfl_sync(mask, g, src);
        int same = __all_sync(mask, g == g0);

        scalar_t s_t, invs_t, q0_t;
        if (same) {
            if (lane == src) {
                decode_qmeta_to_wdtype<scalar_t>(&qmeta[row * G + g0], bits, s_t, invs_t, q0_t);
            }
            // broadcast as 32-bit lanes; scalar_t is 16/32-bit, so broadcast via uint32
            uint32_t s_u  = __shfl_sync(mask, *reinterpret_cast<uint32_t*>(&s_t),  src);
            uint32_t is_u = __shfl_sync(mask, *reinterpret_cast<uint32_t*>(&invs_t), src);
            uint32_t q0_u = __shfl_sync(mask, *reinterpret_cast<uint32_t*>(&q0_t), src);
            s_t    = *reinterpret_cast<scalar_t*>(&s_u);
            invs_t = *reinterpret_cast<scalar_t*>(&is_u);
            q0_t   = *reinterpret_cast<scalar_t*>(&q0_u);
        } else {
            decode_qmeta_to_wdtype<scalar_t>(&qmeta[row * G + g], bits, s_t, invs_t, q0_t);
        }

        scalar_t err_t, deq_t;
        uint8_t qb;
        quantize_scalar_wdtype<scalar_t>(x[t], invs_t, s_t, q0_t, maxq_i, err_t, qb, deq_t);

        // Commit
        qweight[row * R + r] = qb;
        W[row * R + r]       = deq_t;
        Eblk[t * R + r]      = err_t;

        // Propagate to earlier rows i < t: x[i] += S(i,t) * err_t
#pragma unroll
        for (int i = 0; i < MAX_B; ++i) {
            if (i < t) {
                scalar_t alpha = S_sh[i * MAX_B + t];
                x[i] = WOps<scalar_t>::fma(alpha, err_t, x[i]);
            }
        }
    }
}

// ============================================================================
// Kernel 2: Prefix update in pure W dtype ops (no addmm_ / GEMM).
// W_left[i,r] += sum_k (A[i, block_start+k] * invD[i]) * Eblk[k,r]
// ============================================================================
template <typename scalar_t, int MAX_B, int TILE_R, int TILE_I>
__global__ void babai_update_left_kernel_wdtype(
    scalar_t* __restrict__ W,            // [C, R]
    const scalar_t* __restrict__ A,      // [C, C]
    const scalar_t* __restrict__ invD,   // [C]
    const scalar_t* __restrict__ Eblk,   // [B, R]
    int C, int R,
    int block_start, int B
) {
    __shared__ scalar_t E_sh[MAX_B * TILE_R];   // [B][TILE_R]
    __shared__ scalar_t S_sh[TILE_I * MAX_B];   // [TILE_I][B]

    int r0 = blockIdx.x * TILE_R;
    int i0 = blockIdx.y * TILE_I;

    int tx = threadIdx.x; // 0..TILE_R-1
    int ty = threadIdx.y; // 0..TILE_I-1

    // load E tile: E_sh[k, tx]
    for (int k = ty; k < B; k += TILE_I) {
        int r = r0 + tx;
        E_sh[k * TILE_R + tx] = (r < R) ? Eblk[k * R + r] : WOps<scalar_t>::zero();
    }

    // load S tile: S_sh[ty, k] for k in [0,B)
    int i = i0 + ty;
    if (i < block_start) {
        scalar_t invd = invD[i];
        for (int k = tx; k < B; k += TILE_R) {
            scalar_t a = A[i * C + (block_start + k)];
            S_sh[ty * MAX_B + k] = WOps<scalar_t>::mul(a, invd);
        }
    } else {
        for (int k = tx; k < B; k += TILE_R) {
            S_sh[ty * MAX_B + k] = WOps<scalar_t>::zero();
        }
    }

    __syncthreads();

    int r = r0 + tx;
    if (i < block_start && r < R) {
        scalar_t acc = WOps<scalar_t>::zero();
#pragma unroll
        for (int k = 0; k < MAX_B; ++k) {
            if (k < B) {
                acc = WOps<scalar_t>::fma(S_sh[ty * MAX_B + k], E_sh[k * TILE_R + tx], acc);
            }
        }
        W[i * R + r] = WOps<scalar_t>::add(W[i * R + r], acc);
    }
}
}

// ============================================================================
// Host wrapper
// ============================================================================
torch::Tensor babai_solver_cuda(
    torch::Tensor weight,      // [C, R]
    torch::Tensor A,           // [C, C] upper-tri = chol(H)^T
    torch::Tensor qmeta_bytes, // [C, G, 4] or [C*G, 4]
    int64_t group_size,
    int64_t bits,
    int64_t block_size
) {
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(A.is_cuda(),      "A must be CUDA");
    TORCH_CHECK(qmeta_bytes.is_cuda(), "qmeta_bytes must be CUDA");

    weight      = weight.contiguous();
    //A           = A.contiguous();
    qmeta_bytes = qmeta_bytes.contiguous();

    const int64_t C = weight.size(0);
    const int64_t R = weight.size(1);

    TORCH_CHECK(weight.dim() == 2, "weight must be [C, R]");
    TORCH_CHECK(A.dim() == 2 && A.size(0) == C && A.size(1) == C, "A must be [C, C]");

    // Determine G
    int64_t G;
    if (qmeta_bytes.dim() == 3) {
        TORCH_CHECK(qmeta_bytes.size(0) == C, "qmeta_bytes[0] must be C");
        TORCH_CHECK(qmeta_bytes.size(2) == 4, "qmeta_bytes[...,4] expected");
        G = qmeta_bytes.size(1);
    } else {
        TORCH_CHECK(qmeta_bytes.dim() == 2 && qmeta_bytes.size(1) == 4,
                    "qmeta_bytes must be [C,G,4] or [C*G,4]");
        TORCH_CHECK(qmeta_bytes.size(0) % C == 0, "qmeta_bytes[0] must be multiple of C");
        G = qmeta_bytes.size(0) / C;
    }

    constexpr int MAX_B = 32;
    if (block_size <= 0 || block_size > MAX_B) block_size = MAX_B;
    if (block_size > C) block_size = C;

    auto st  = weight.scalar_type();
    auto dev = weight.device();

    auto qweight = torch::empty({C, R},
        torch::TensorOptions().dtype(torch::kUInt8).device(dev));

    // Eblk in W dtype
    auto Eblk = torch::empty({block_size, R},
        torch::TensorOptions().dtype(st).device(dev));

    auto qmeta_flat =
        (qmeta_bytes.dim() == 3) ? qmeta_bytes.view({C * G, 4}) : qmeta_bytes;
    const QMetaPacked* qmeta_ptr =
        reinterpret_cast<const QMetaPacked*>(qmeta_flat.data_ptr<uint8_t>());

    // A cast to W dtype (no fp32 A_f)
    A  = (A.scalar_type() == st) ? A : A.to(st);
    A = A.contiguous();

    // invD_all in W dtype
    auto invD_all = A.diagonal(0, 0, 1).reciprocal().contiguous();

    auto stream = at::cuda::getCurrentCUDAStream();

    constexpr int THREADS_Q = 128;
    constexpr int TILE_R = 64;
    constexpr int TILE_I = 4;
    dim3 upd_block(TILE_R, TILE_I);

    // Right-to-left blocks
    for (int64_t block_end = C; block_end > 0; block_end -= block_size) {
        const int64_t block_start = std::max<int64_t>(0, block_end - block_size);
        const int64_t B_long      = block_end - block_start;
        const int     B           = static_cast<int>(B_long);

        auto Eblk_view = Eblk.narrow(0, 0, B_long); // [B, R]

        // 1) Quantize block + intra-block propagation
        const int grid_q = (static_cast<int>(R) + THREADS_Q - 1) / THREADS_Q;

        if (st == at::ScalarType::Float) {
            babai_quant_block_kernel_wdtype<float, MAX_B><<<grid_q, THREADS_Q, 0, stream>>>(
                weight.data_ptr<float>(),
                qweight.data_ptr<uint8_t>(),
                qmeta_ptr,
                A.data_ptr<float>(),
                invD_all.data_ptr<float>(),
                Eblk_view.data_ptr<float>(),
                (int)C, (int)R, (int)G,
                (int)block_start, B,
                (int)group_size,
                (uint8_t)bits
            );
        } else if (st == at::ScalarType::Half) {
            babai_quant_block_kernel_wdtype<__half, MAX_B><<<grid_q, THREADS_Q, 0, stream>>>(
                reinterpret_cast<__half*>(weight.data_ptr<at::Half>()),
                qweight.data_ptr<uint8_t>(),
                qmeta_ptr,
                reinterpret_cast<__half*>(A.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(invD_all.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(Eblk_view.data_ptr<at::Half>()),
                (int)C, (int)R, (int)G,
                (int)block_start, B,
                (int)group_size,
                (uint8_t)bits
            );
        } else if (st == at::ScalarType::BFloat16) {
            babai_quant_block_kernel_wdtype<__nv_bfloat16, MAX_B><<<grid_q, THREADS_Q, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
                qweight.data_ptr<uint8_t>(),
                qmeta_ptr,
                reinterpret_cast<__nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(invD_all.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(Eblk_view.data_ptr<at::BFloat16>()),
                (int)C, (int)R, (int)G,
                (int)block_start, B,
                (int)group_size,
                (uint8_t)bits
            );
        } else {
            TORCH_CHECK(false, "Unsupported dtype for wdtype Babai solver (expected float/half/bfloat16)");
        }
        CUDA_CHECK(cudaGetLastError());

        // 2) Prefix update (rows [0, block_start)) using pure W dtype ops
        if (block_start > 0) {
            int grid_x = (static_cast<int>(R) + TILE_R - 1) / TILE_R;
            int grid_y = (static_cast<int>(block_start) + TILE_I - 1) / TILE_I;
            dim3 upd_grid(grid_x, grid_y);

            if (st == at::ScalarType::Float) {
                babai_update_left_kernel_wdtype<float, MAX_B, TILE_R, TILE_I><<<upd_grid, upd_block, 0, stream>>>(
                    weight.data_ptr<float>(),
                    A.data_ptr<float>(),
                    invD_all.data_ptr<float>(),
                    Eblk_view.data_ptr<float>(),
                    (int)C, (int)R,
                    (int)block_start, B
                );
            } else if (st == at::ScalarType::Half) {
                babai_update_left_kernel_wdtype<__half, MAX_B, TILE_R, TILE_I><<<upd_grid, upd_block, 0, stream>>>(
                    reinterpret_cast<__half*>(weight.data_ptr<at::Half>()),
                    reinterpret_cast<__half*>(A.data_ptr<at::Half>()),
                    reinterpret_cast<__half*>(invD_all.data_ptr<at::Half>()),
                    reinterpret_cast<__half*>(Eblk_view.data_ptr<at::Half>()),
                    (int)C, (int)R,
                    (int)block_start, B
                );
            } else { // BFloat16
                babai_update_left_kernel_wdtype<__nv_bfloat16, MAX_B, TILE_R, TILE_I><<<upd_grid, upd_block, 0, stream>>>(
                    reinterpret_cast<__nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
                    reinterpret_cast<__nv_bfloat16*>(A_data_ptr<at::BFloat16>()),
                    reinterpret_cast<__nv_bfloat16*>(invD_all.data_ptr<at::BFloat16>()),
                    reinterpret_cast<__nv_bfloat16*>(Eblk_view.data_ptr<at::BFloat16>()),
                    (int)C, (int)R,
                    (int)block_start, B
                );
            }
            CUDA_CHECK(cudaGetLastError());
        }
    }

    return qweight;
}