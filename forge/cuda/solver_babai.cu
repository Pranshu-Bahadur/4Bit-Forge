// solver_babai.cu (wdtype version, updated for NEW qmeta contract)
//
// NEW contract:
// - W is transposed: [C, R] where C=in_features, R=out_features
// - qmeta is per-output, grouped along input axis:
//     qmeta shape [R, G, 4] (or flat [R*G, 4])
//     G = ceil(C / group_size)
// - For a given (row=input idx, r=output idx):
//     g = row / group_size
//     qmeta entry = qmeta[r * G + g]
//
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
    __device__ __forceinline__ static T fma(T a, T b, T c) { return __hfma(a, b, c); }
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
        // keep it in bf16 ops (mul+add)
        return add(mul(a, b), c);
    }
    __device__ __forceinline__ static float to_f32(T a) { return __bfloat162float(a); }
    __device__ __forceinline__ static T from_f32(float x) { return __float2bfloat16_rn(x); }
};

// -----------------------------
// Safe shuffle-broadcast for scalar_t (avoid illegal 32-bit loads on 16-bit types)
// (kept for API compatibility; not used for qmeta under the NEW contract)
// -----------------------------
__device__ __forceinline__ float shfl_broadcast(unsigned mask, float v, int src) {
    return __shfl_sync(mask, v, src);
}
__device__ __forceinline__ __half shfl_broadcast(unsigned mask, __half v, int src) {
    __half_raw hr = *reinterpret_cast<const __half_raw*>(&v);
    uint32_t u = static_cast<uint32_t>(hr.x);
    u = __shfl_sync(mask, u, src);
    __half_raw out; out.x = static_cast<uint16_t>(u);
    return *reinterpret_cast<__half*>(&out);
}
__device__ __forceinline__ __nv_bfloat16 shfl_broadcast(unsigned mask, __nv_bfloat16 v, int src) {
    __nv_bfloat16_raw br = *reinterpret_cast<const __nv_bfloat16_raw*>(&v);
    uint32_t u = static_cast<uint32_t>(br.x);
    u = __shfl_sync(mask, u, src);
    __nv_bfloat16_raw out; out.x = static_cast<uint16_t>(u);
    return *reinterpret_cast<__nv_bfloat16*>(&out);
}

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
// quantize step: computations in W dtype; rounding uses fp32 view of biased
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
    // biased = x * inv_s + q0 (W dtype)
    scalar_t biased_t = WOps<scalar_t>::fma(x, inv_s, q0);

    // rounding decision needs fp32
    float biased_f = WOps<scalar_t>::to_f32(biased_t);
    int q = __float2int_rn(biased_f);
    q = (q < 0) ? 0 : q;
    q = (q > maxq) ? maxq : q;

    // deq = (q - q0) * s (W dtype)
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
//
// NEW qmeta: per-output r, per-input-group g => qmeta[r * G + g]
// g is computed from the input-feature index (row).
// ============================================================================
template <typename scalar_t, int MAX_B>
__global__ void babai_quant_block_kernel_wdtype(
    scalar_t* __restrict__ W,                   // [C, R]
    uint8_t*  __restrict__ qweight,             // [C, R]
    const QMetaPacked* __restrict__ qmeta,      // [R * G]
    const scalar_t* __restrict__ A,             // [C, C] upper-tri (W dtype)
    const scalar_t* __restrict__ invD_all,      // [C] invD = 1/Aii (W dtype)
    scalar_t* __restrict__ Eblk,                // [B, R] (W dtype)
    int C, int R, int G,
    int block_start, int B,
    int group_size,
    uint8_t bits
) {
    __shared__ scalar_t S_sh[MAX_B * MAX_B];

    // Build row-scaled block: S(i,t) = A(i,t) * invD(i) for t>i
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
    if (r >= R) return;

    

    // Load x[0:B] = W[block_start:block_start+B, r]
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

        // groups along input dim (C)
        int g = row / group_size;
        if (g >= G) g = G - 1; // safety (shouldn't trigger if host checks are correct)

        scalar_t s_t, invs_t, q0_t;
        // NEW indexing: qmeta[r * G + g]
        decode_qmeta_to_wdtype<scalar_t>(&qmeta[r * G + g], bits, s_t, invs_t, q0_t);

        scalar_t err_t, deq_t;
        uint8_t qb;
        quantize_scalar_wdtype<scalar_t>(x[t], invs_t, s_t, q0_t, maxq_i, err_t, qb, deq_t);

        qweight[row * R + r] = qb;
        W[row * R + r]       = deq_t;   // commits in W.dtype world
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
// fill_A_scaled_kernel: build A_view(i,k) = A(i, block_start+k) * invD(i)
// into A_tmp laid out [C, block_size] (we only write i < block_start, k < B).
// ============================================================================
template <typename scalar_t>
__global__ void fill_A_scaled_kernel(
    scalar_t* __restrict__ A_tmp,         // [C, block_size]
    const scalar_t* __restrict__ A,       // [C, C]
    const scalar_t* __restrict__ invD,    // [C]
    int C, int block_size,
    int block_start, int B
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x; // 0..B-1
    int i = blockIdx.y * blockDim.y + threadIdx.y; // 0..block_start-1
    if (i >= block_start || k >= B) return;

    scalar_t a    = A[i * C + (block_start + k)];
    scalar_t invd = invD[i];
    A_tmp[i * block_size + k] = WOps<scalar_t>::mul(a, invd);
}

} // namespace

// ============================================================================
// Host wrapper (updated for NEW qmeta contract)
// ============================================================================
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

    weight      = weight.contiguous();
    qmeta_bytes = qmeta_bytes.contiguous();

    TORCH_CHECK(weight.dim() == 2, "weight must be [C, R]");
    const int64_t C = weight.size(0);
    const int64_t R = weight.size(1);

    TORCH_CHECK(A.dim() == 2 && A.size(0) == C && A.size(1) == C, "A must be [C, C]");

    // Determine G from qmeta
    int64_t G;
    if (qmeta_bytes.dim() == 3) {
        TORCH_CHECK(qmeta_bytes.size(0) == R, "qmeta_bytes[0] must be R (out_features)");
        TORCH_CHECK(qmeta_bytes.size(2) == 4, "qmeta_bytes[...,4] expected");
        G = qmeta_bytes.size(1);
    } else {
        TORCH_CHECK(qmeta_bytes.dim() == 2 && qmeta_bytes.size(1) == 4,
                    "qmeta_bytes must be [R,G,4] or [R*G,4]");
        TORCH_CHECK(qmeta_bytes.size(0) % R == 0, "qmeta_bytes[0] must be multiple of R");
        G = qmeta_bytes.size(0) / R;
    }

    // Effective group_size is along C
    TORCH_CHECK(group_size > 0, "group_size must be > 0");
    if (group_size > C) group_size = C;
    TORCH_CHECK(group_size % 32 == 0, "group_size must be multiple of 32");

    const int64_t expected_G = (C + group_size - 1) / group_size;
    TORCH_CHECK(expected_G == G,
                "qmeta G mismatch: got G=", G,
                " expected ceil(C/group_size)=", expected_G,
                " with C=", C, " group_size=", group_size);

    constexpr int MAX_B = 32;
    if (block_size <= 0 || block_size > MAX_B) block_size = MAX_B;
    if (block_size > C) block_size = C;

    auto st  = weight.scalar_type();
    auto dev = weight.device();

    auto qweight = torch::zeros({C, R},
        torch::TensorOptions().dtype(torch::kUInt8).device(dev));

    // Eblk in W dtype
    auto Eblk = torch::empty({block_size, R},
        torch::TensorOptions().dtype(st).device(dev));

    auto qmeta_flat = (qmeta_bytes.dim() == 3) ? qmeta_bytes.view({R * G, 4}) : qmeta_bytes;
    const QMetaPacked* qmeta_ptr =
        reinterpret_cast<const QMetaPacked*>(qmeta_flat.data_ptr<uint8_t>());

    // A cast to W dtype
    torch::Tensor A_t = (A.scalar_type() == st) ? A : A.to(st);
    A_t = A_t.contiguous();

    // invD_all in W dtype
    auto invD_all = A_t.diagonal(0, 0, 1).reciprocal().contiguous();

    // Scratch for addmm_ prefix update (valid for fp16/bf16/fp32)
    torch::Tensor A_tmp;
    if (st == at::ScalarType::Float || st == at::ScalarType::Half || st == at::ScalarType::BFloat16) {
        A_tmp = torch::empty({C, block_size},
            torch::TensorOptions().dtype(st).device(dev));
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int THREADS_Q = 128;

    // fill_A_scaled launch config
    dim3 blk2d(16, 16);

    // Right-to-left blocks along C
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
                A_t.data_ptr<float>(),
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
                reinterpret_cast<__half*>(A_t.data_ptr<at::Half>()),
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
                reinterpret_cast<__nv_bfloat16*>(A_t.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(invD_all.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(Eblk_view.data_ptr<at::BFloat16>()),
                (int)C, (int)R, (int)G,
                (int)block_start, B,
                (int)group_size,
                (uint8_t)bits
            );
        } else {
            TORCH_CHECK(false, "Unsupported dtype (expected float/half/bfloat16)");
        }
        CUDA_CHECK(cudaGetLastError());

        // 2) Prefix update (rows [0, block_start)) using addmm_ for fp16/bf16/fp32
        if (block_start > 0) {
            dim3 grd2d((B + blk2d.x - 1) / blk2d.x,
                       ((int)block_start + blk2d.y - 1) / blk2d.y);

            if (st == at::ScalarType::Float) {
                fill_A_scaled_kernel<float><<<grd2d, blk2d, 0, stream>>>(
                    A_tmp.data_ptr<float>(),
                    A_t.data_ptr<float>(),
                    invD_all.data_ptr<float>(),
                    (int)C, (int)block_size,
                    (int)block_start, B
                );
                CUDA_CHECK(cudaGetLastError());

                auto W_left = weight.narrow(0, 0, block_start); // [block_start, R]
                auto A_view = A_tmp.narrow(0, 0, block_start).narrow(1, 0, B); // [block_start, B]
                W_left.addmm_(A_view, Eblk_view, /*beta=*/1.0, /*alpha=*/1.0);
            } else if (st == at::ScalarType::Half) {
                fill_A_scaled_kernel<__half><<<grd2d, blk2d, 0, stream>>>(
                    reinterpret_cast<__half*>(A_tmp.data_ptr<at::Half>()),
                    reinterpret_cast<__half*>(A_t.data_ptr<at::Half>()),
                    reinterpret_cast<__half*>(invD_all.data_ptr<at::Half>()),
                    (int)C, (int)block_size,
                    (int)block_start, B
                );
                CUDA_CHECK(cudaGetLastError());

                auto W_left = weight.narrow(0, 0, block_start);
                auto A_view = A_tmp.narrow(0, 0, block_start).narrow(1, 0, B);
                W_left.addmm_(A_view, Eblk_view, /*beta=*/1.0, /*alpha=*/1.0);
            } else if (st == at::ScalarType::BFloat16) {
                fill_A_scaled_kernel<__nv_bfloat16><<<grd2d, blk2d, 0, stream>>>(
                    reinterpret_cast<__nv_bfloat16*>(A_tmp.data_ptr<at::BFloat16>()),
                    reinterpret_cast<__nv_bfloat16*>(A_t.data_ptr<at::BFloat16>()),
                    reinterpret_cast<__nv_bfloat16*>(invD_all.data_ptr<at::BFloat16>()),
                    (int)C, (int)block_size,
                    (int)block_start, B
                );
                CUDA_CHECK(cudaGetLastError());

                auto W_left = weight.narrow(0, 0, block_start);
                auto A_view = A_tmp.narrow(0, 0, block_start).narrow(1, 0, B);
                W_left.addmm_(A_view, Eblk_view, /*beta=*/1.0, /*alpha=*/1.0);
            } else {
                TORCH_CHECK(false, "Unexpected dtype in addmm_ path");
            }
        }
    }

    return qweight;
}