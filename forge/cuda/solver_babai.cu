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

// ---------------- dtype helpers (global storage stays in scalar_t) ----------------

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

template <typename scalar_t>
__device__ __forceinline__ scalar_t madd_scalar(scalar_t x, scalar_t a, scalar_t b) {
    // x + a*b in the "W dtype world" (storage/rounding in scalar_t)
    float xf = to_f32(x);
    float af = to_f32(a);
    float bf = to_f32(b);
    return from_f32<scalar_t>(__fmaf_rn(af, bf, xf));
}

// ---------------- qmeta decode (float math -> used by quant primitive) ---------------

__device__ __forceinline__ void decode_qmeta_f32(
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

// ---------------- quant primitive: returns err/deq committed into scalar_t -----------

template <typename scalar_t>
__device__ __forceinline__ void quantize_scalar_wdtype(
    scalar_t x_t,
    float inv_s, float s, float q0, int maxq_i,
    scalar_t& err_t_out, uint8_t& q_out, scalar_t& deq_t_out
) {
    float x = to_f32(x_t);
    float biased = __fmaf_rn(x, inv_s, q0);
    int q = __float2int_rn(biased);

    q = (q < 0) ? 0 : q;
    q = (q > maxq_i) ? maxq_i : q;

    float deq = __fmaf_rn(static_cast<float>(q), s, -q0 * s);
    float err = x - deq;

    deq_t_out = from_f32<scalar_t>(deq);
    err_t_out = from_f32<scalar_t>(err);
    q_out     = static_cast<uint8_t>(q);
}

// -----------------------------------------------------------------------------
// Scratch fill kernel for addmm_ path (ALL inputs/outputs in scalar_t)
// A_tmp[i,k] = A[i, block_start+k] * invD[i]
// -----------------------------------------------------------------------------

template <typename scalar_t>
__global__ void fill_A_scaled_kernel_wdtype(
    scalar_t* __restrict__ A_tmp,      // [C, block_size] row-major
    const scalar_t* __restrict__ A,    // [C, C] in W dtype
    const scalar_t* __restrict__ invD, // [C]   in W dtype
    int C, int block_size,
    int block_start, int B
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x; // 0..B-1
    int i = blockIdx.y * blockDim.y + threadIdx.y; // 0..block_start-1
    if (k >= B || i >= block_start) return;

    float a = to_f32(A[i * C + (block_start + k)]);
    float d = to_f32(invD[i]);
    float v = a * d;
    A_tmp[i * block_size + k] = from_f32<scalar_t>(v);
}

// -----------------------------------------------------------------------------
// Kernel 1: Babai quantize block (back-to-front) + intra-block propagation.
// EVERYTHING stored/propagated in scalar_t (Eblk is scalar_t).
// -----------------------------------------------------------------------------

template <typename scalar_t, int MAX_B>
__global__ void babai_quant_block_kernel_wdtype(
    scalar_t* __restrict__ W,                   // [C, R] (mutable, W dtype)
    uint8_t*  __restrict__ qweight,             // [C, R] uint8
    const QMetaPacked* __restrict__ qmeta,      // [C * G]
    const scalar_t* __restrict__ A,             // [C, C] (W dtype, upper-tri expected)
    const scalar_t* __restrict__ invD_all,      // [C]    (W dtype) invD[i] = 1/A[i,i]
    scalar_t* __restrict__ Eblk,                // [B, R] (W dtype)
    int C, int R, int G,
    int block_start, int B,
    int group_size,
    uint8_t bits
) {
    __shared__ scalar_t S_sh[MAX_B * MAX_B];

    // Build S_sh(i,t) = A(i,t) * invD(i) for t>i, else 0 (all in W dtype storage)
    for (int idx = threadIdx.x; idx < B * B; idx += blockDim.x) {
        int i = idx / B;
        int t = idx - i * B;

        scalar_t v = from_f32<scalar_t>(0.0f);
        if (t > i) {
            int gi = block_start + i;
            int gt = block_start + t;
            float a = to_f32(A[gi * C + gt]);
            float d = to_f32(invD_all[gi]);
            v = from_f32<scalar_t>(a * d);
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

    // Load column (in W dtype)
    scalar_t x[MAX_B];
    #pragma unroll
    for (int i = 0; i < MAX_B; ++i) x[i] = from_f32<scalar_t>(0.0f);

    for (int i = 0; i < B; ++i) {
        int row = block_start + i;
        x[i] = W[row * R + r];
    }

    const int maxq_i = (1 << bits) - 1;

    // Back-to-front within block
    for (int t = B - 1; t >= 0; --t) {
        int row = block_start + t;

        // Group id per lane (same optimization as before)
        int g  = r / group_size;
        int g0 = __shfl_sync(mask, g, src);
        int same = __all_sync(mask, g == g0);

        float s = 0.f, inv_s = 0.f, q0 = 0.f;
        if (same) {
            if (lane == src) {
                uint32_t packed = qmeta_to_u32(qmeta[row * G + g0]);
                decode_qmeta_f32(packed, s, inv_s, q0, bits);
            }
            s     = __shfl_sync(mask, s, src);
            inv_s = __shfl_sync(mask, inv_s, src);
            q0    = __shfl_sync(mask, q0, src);
        } else {
            uint32_t packed = qmeta_to_u32(qmeta[row * G + g]);
            decode_qmeta_f32(packed, s, inv_s, q0, bits);
        }

        scalar_t err_t, deq_t;
        uint8_t qb;
        quantize_scalar_wdtype<scalar_t>(x[t], inv_s, s, q0, maxq_i, err_t, qb, deq_t);

        // Commit
        qweight[row * R + r] = qb;
        W[row * R + r]       = deq_t;
        Eblk[t * R + r]      = err_t;

        // Propagate to earlier rows i < t in the "W dtype world"
        #pragma unroll
        for (int i = 0; i < MAX_B; ++i) {
            if (i < t) {
                scalar_t alpha = S_sh[i * MAX_B + t];
                x[i] = madd_scalar<scalar_t>(x[i], alpha, err_t);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel 2: Prefix update (generic). Storage in scalar_t.
// W[i,r] += sum_k S(i,k) * Eblk[k,r], where S(i,k)=A(i,block_start+k)*invD(i)
// -----------------------------------------------------------------------------

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

    // load E tile (W dtype)
    for (int k = ty; k < B; k += TILE_I) {
        int r = r0 + tx;
        E_sh[k * TILE_R + tx] = (r < R) ? Eblk[k * R + r] : from_f32<scalar_t>(0.0f);
    }

    // load S tile (W dtype)
    int i = i0 + ty;
    if (i < block_start) {
        float invd = to_f32(invD[i]);
        for (int k = tx; k < B; k += TILE_R) {
            float a = to_f32(A[i * C + (block_start + k)]);
            S_sh[ty * MAX_B + k] = from_f32<scalar_t>(a * invd);
        }
    } else {
        for (int k = tx; k < B; k += TILE_R) {
            S_sh[ty * MAX_B + k] = from_f32<scalar_t>(0.0f);
        }
    }

    __syncthreads();

    int r = r0 + tx;
    if (i < block_start && r < R) {
        scalar_t acc = from_f32<scalar_t>(0.0f);
        #pragma unroll
        for (int k = 0; k < MAX_B; ++k) {
            if (k < B) {
                acc = madd_scalar<scalar_t>(acc, S_sh[ty * MAX_B + k], E_sh[k * TILE_R + tx]);
            }
        }
        W[i * R + r] = madd_scalar<scalar_t>(W[i * R + r], from_f32<scalar_t>(1.0f), acc);
    }
}

} // namespace

// -----------------------------------------------------------------------------
// Host wrapper
// -----------------------------------------------------------------------------
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

    auto st = weight.scalar_type();
    auto dev = weight.device();

    // Keep outputs/scratch in the "W dtype world"
    auto qweight = torch::zeros({C, R},
        torch::TensorOptions().dtype(torch::kUInt8).device(dev));

    auto Eblk = torch::zeros({block_size, R},
        torch::TensorOptions().dtype(st).device(dev));

    auto qmeta_flat =
        (qmeta_bytes.dim() == 3) ? qmeta_bytes.view({C * G, 4}) : qmeta_bytes;

    // Cast A to W dtype (no fp32 A_f)
    A = (A.scalar_type() == st) ? A : A.to(st);
    A = A.contiguous();

    // invD_all = 1 / diag(A) (in W dtype)
    auto invD_all = A.diagonal(0, 0, 1).reciprocal().contiguous();

    // Scratch for addmm_ path (in W dtype)
    torch::Tensor A_tmp = torch::zeros({C, block_size},
        torch::TensorOptions().dtype(st).device(dev));

    auto stream = at::cuda::getCurrentCUDAStream();

    constexpr int THREADS_Q = 128;
    constexpr int TILE_R = 64;
    constexpr int TILE_I = 4;
    dim3 tile_block(TILE_R, TILE_I);

    // Right-to-left blocks
    for (int64_t block_end = C; block_end > 0; block_end -= block_size) {
        const int64_t block_start = std::max<int64_t>(0, block_end - block_size);
        const int64_t B_long      = block_end - block_start;
        const int     B           = static_cast<int>(B_long);

        auto Eblk_view = Eblk.narrow(0, 0, B_long); // [B, R] (W dtype)

        // 1) Quantize block + intra-block propagation
        const int grid_q = (static_cast<int>(R) + THREADS_Q - 1) / THREADS_Q;

        if (st == at::ScalarType::Float) {
            babai_quant_block_kernel_wdtype<float, MAX_B><<<grid_q, THREADS_Q, 0, stream>>>(
                weight.data_ptr<float>(),
                qweight.data_ptr<uint8_t>(),
                reinterpret_cast<const QMetaPacked*>(qmeta_flat.data_ptr<uint8_t>()),
                A.data_ptr<float>(),
                invD_all.data_ptr<float>(),
                (float*)Eblk_view.data_ptr<float>(),
                (int)C, (int)R, (int)G,
                (int)block_start, B,
                (int)group_size,
                (uint8_t)bits
            );
        } else if (st == at::ScalarType::Half) {
            babai_quant_block_kernel_wdtype<at::Half, MAX_B><<<grid_q, THREADS_Q, 0, stream>>>(
                (at::Half*)weight.data_ptr<at::Half>(),
                qweight.data_ptr<uint8_t>(),
                reinterpret_cast<const QMetaPacked*>(qmeta_flat.data_ptr<uint8_t>()),
                (at::Half*)A.data_ptr<at::Half>(),
                (at::Half*)invD_all.data_ptr<at::Half>(),
                (at::Half*)Eblk_view.data_ptr<at::Half>(),
                (int)C, (int)R, (int)G,
                (int)block_start, B,
                (int)group_size,
                (uint8_t)bits
            );
        } else if (st == at::ScalarType::BFloat16) {
            babai_quant_block_kernel_wdtype<at::BFloat16, MAX_B><<<grid_q, THREADS_Q, 0, stream>>>(
                (at::BFloat16*)weight.data_ptr<at::BFloat16>(),
                qweight.data_ptr<uint8_t>(),
                reinterpret_cast<const QMetaPacked*>(qmeta_flat.data_ptr<uint8_t>()),
                (at::BFloat16*)A.data_ptr<at::BFloat16>(),
                (at::BFloat16*)invD_all.data_ptr<at::BFloat16>(),
                (at::BFloat16*)Eblk_view.data_ptr<at::BFloat16>(),
                (int)C, (int)R, (int)G,
                (int)block_start, B,
                (int)group_size,
                (uint8_t)bits
            );
        } else if (st == at::ScalarType::Float8_e4m3fn) {
            babai_quant_block_kernel_wdtype<c10::Float8_e4m3fn, MAX_B><<<grid_q, THREADS_Q, 0, stream>>>(
                (c10::Float8_e4m3fn*)weight.data_ptr<c10::Float8_e4m3fn>(),
                qweight.data_ptr<uint8_t>(),
                reinterpret_cast<const QMetaPacked*>(qmeta_flat.data_ptr<uint8_t>()),
                (c10::Float8_e4m3fn*)A.data_ptr<c10::Float8_e4m3fn>(),
                (c10::Float8_e4m3fn*)invD_all.data_ptr<c10::Float8_e4m3fn>(),
                (c10::Float8_e4m3fn*)Eblk_view.data_ptr<c10::Float8_e4m3fn>(),
                (int)C, (int)R, (int)G,
                (int)block_start, B,
                (int)group_size,
                (uint8_t)bits
            );
        } else {
            TORCH_CHECK(false, "Unsupported dtype for babai_solver_cuda (wdtype version)");
        }
        CUDA_CHECK(cudaGetLastError());

        // 2) Prefix update (rows [0, block_start))
        if (block_start > 0) {
            // Option A: fully W-dtype tiled kernel (also works for fp8)
            int grid_x = (static_cast<int>(R) + TILE_R - 1) / TILE_R;
            int grid_y = (static_cast<int>(block_start) + TILE_I - 1) / TILE_I;
            dim3 grid(grid_x, grid_y); // <-- (typo guard below)

            (void)grid; // silence unused if we switch paths

            // Option B (fp16/bf16/fp32): addmm_ in W dtype (A_tmp + Eblk in W dtype)
            if (st == at::ScalarType::Float || st == at::ScalarType::Half || st == at::ScalarType::BFloat16) {
                const int tx = 16, ty = 16;
                dim3 blk(tx, ty);
                dim3 grd((B + tx - 1) / tx, ((int)block_start + ty - 1) / ty);

                if (st == at::ScalarType::Float) {
                    fill_A_scaled_kernel_wdtype<float><<<grd, blk, 0, stream>>>(
                        A_tmp.data_ptr<float>(),
                        A.data_ptr<float>(),
                        invD_all.data_ptr<float>(),
                        (int)C, (int)block_size,
                        (int)block_start, B
                    );
                    CUDA_CHECK(cudaGetLastError());

                    auto W_left = weight.narrow(0, 0, block_start);
                    auto A_view = A_tmp.narrow(0, 0, block_start).narrow(1, 0, B_long);
                    W_left.addmm_(A_view, Eblk_view, /*beta=*/1.0, /*alpha=*/1.0);
                } else if (st == at::ScalarType::Half) {
                    fill_A_scaled_kernel_wdtype<at::Half><<<grd, blk, 0, stream>>>(
                        (at::Half*)A_tmp.data_ptr<at::Half>(),
                        (at::Half*)A.data_ptr<at::Half>(),
                        (at::Half*)invD_all.data_ptr<at::Half>(),
                        (int)C, (int)block_size,
                        (int)block_start, B
                    );
                    CUDA_CHECK(cudaGetLastError());

                    auto W_left = weight.narrow(0, 0, block_start);
                    auto A_view = A_tmp.narrow(0, 0, block_start).narrow(1, 0, B_long);
                    W_left.addmm_(A_view, Eblk_view, /*beta=*/1.0, /*alpha=*/1.0);
                } else { // BFloat16
                    fill_A_scaled_kernel_wdtype<at::BFloat16><<<grd, blk, 0, stream>>>(
                        (at::BFloat16*)A_tmp.data_ptr<at::BFloat16>(),
                        (at::BFloat16*)A.data_ptr<at::BFloat16>(),
                        (at::BFloat16*)invD_all.data_ptr<at::BFloat16>(),
                        (int)C, (int)block_size,
                        (int)block_start, B
                    );
                    CUDA_CHECK(cudaGetLastError());

                    auto W_left = weight.narrow(0, 0, block_start);
                    auto A_view = A_tmp.narrow(0, 0, block_start).narrow(1, 0, B_long);
                    W_left.addmm_(A_view, Eblk_view, /*beta=*/1.0, /*alpha=*/1.0);
                }
            } else if (st == at::ScalarType::Float8_e4m3fn) {
                // fp8: use W-dtype tiled kernel update
                int grid_x2 = (static_cast<int>(R) + TILE_R - 1) / TILE_R;
                int grid_y2 = (static_cast<int>(block_start) + TILE_I - 1) / TILE_I;
                dim3 fp8_grid(grid_x2, grid_y2);

                babai_update_left_kernel_wdtype<c10::Float8_e4m3fn, MAX_B, TILE_R, TILE_I>
                    <<<fp8_grid, tile_block, 0, stream>>>(
                        (c10::Float8_e4m3fn*)weight.data_ptr<c10::Float8_e4m3fn>(),
                        (c10::Float8_e4m3fn*)A.data_ptr<c10::Float8_e4m3fn>(),
                        (c10::Float8_e4m3fn*)invD_all.data_ptr<c10::Float8_e4m3fn>(),
                        (c10::Float8_e4m3fn*)Eblk_view.data_ptr<c10::Float8_e4m3fn>(),
                        (int)C, (int)R,
                        (int)block_start, B
                    );
                CUDA_CHECK(cudaGetLastError());
            } else {
                TORCH_CHECK(false, "Unexpected dtype in prefix update");
            }
        }
    }

    return qweight;
}