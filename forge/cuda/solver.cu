// solver_fp32.cu
// fp32-only GPTQ solver (debug-friendly)
//
// Assumptions:
//   * W is float32 CUDA tensor shaped [C, R] (C = GPTQ dimension / in_features, R = out_features)
//   * Hinv is float32 CUDA tensor shaped [C, C] containing the UPPER-triangular factor U such that
//         inv(H) = U^T U
//     Equivalently: U = chol(inv(H)).T  (if chol returns lower)
//   * scales and qzeros are float32 CUDA tensors with numel == R * G,
//     where G = ceil(C / group_size). Layout is [R, G] flattened row-major:
//         idx = r * G + g
//   * qweight is uint8 codes, stored as [C, R] (same layout as W).
//
// This file provides:
//   * gptq_quantize_block_fp32: quantize a contiguous block of rows [block_start, block_start+B)
//                              and write Delta (W - Wq) to Eblk.
//   * gptq_trsm_block_fp32:     forward-substitution solve (U_block^T) E^T = Delta^T, in-place on Eblk.
//   * gptq_solve_fp32:          full loop over blocks including tail update via W_tail.addmm_.

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <cstdint>

// -----------------------------
// Checks / helpers
// -----------------------------

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FP32(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_U8(x) TORCH_CHECK((x).scalar_type() == at::kByte, #x " must be uint8")
#define CHECK_I32(x) TORCH_CHECK((x).scalar_type() == at::kInt, #x " must be int32")

static inline int64_t ceil_div_int64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    // Full mask
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

// Clamp to [0, maxq]
__device__ __forceinline__ int clamp_q(int q, int maxq) {
    q = (q < 0) ? 0 : q;
    q = (q > maxq) ? maxq : q;
    return q;
}

// Quantize a scalar using (s, q0) with clamp.
__device__ __forceinline__ void quantize_scalar_u8(
    float x,
    float s,
    float q0,
    int maxq,
    float &err_out,
    uint8_t &q_out,
    float &deq_out
) {
    // Protect against degenerate s
    float inv_s = 1.0f / (s + 1e-20f);

    // q0 is expected integer-ish; clamp after rounding to nearest
    float q0r = rintf(q0);
    q0r = fminf(fmaxf(q0r, 0.0f), (float)maxq);

    float biased = fmaf(x, inv_s, q0r);
    int q = __float2int_rn(biased);
    q = clamp_q(q, maxq);

    // deq = (q - q0) * s
    deq_out = (float(q) - q0r) * s;
    err_out = x - deq_out;
    q_out   = (uint8_t)q;
}

// -----------------------------
// 1) Quantize block -> qweight + Delta (stored to Eblk)
// -----------------------------

// Each CUDA block handles 1 row (one GPTQ dimension index) within the current B-sized block.
// Threads iterate over r in [0, R).
// Eblk layout: [B, R] row-major, storing Delta for each row-in-block.
__global__ void gptq_quantize_block_kernel_fp32(
    float* __restrict__ W,          // [C, R]
    uint8_t* __restrict__ qweight,  // [C, R]
    float* __restrict__ Eblk,       // [B, R]
    const float* __restrict__ scales, // [R*G]
    const float* __restrict__ qzeros, // [R*G]
    const int32_t* __restrict__ g_idx, // [C] or nullptr
    int C,
    int R,
    int G,
    int block_start,
    int B,
    int group_size,
    int bits
) {
    int t = (int)blockIdx.x; // 0..B-1
    if (t >= B) return;

    int row = block_start + t;
    if (row >= C) return;

    int g = 0;
    if (g_idx) {
        g = (int)g_idx[row];
    } else {
        g = row / group_size;
    }
    if (g < 0) g = 0;
    if (g >= G) g = G - 1;

    const int maxq = (1 << bits) - 1;

    float* W_row = W + (int64_t)row * R;
    uint8_t* Q_row = qweight + (int64_t)row * R;
    float* E_row = Eblk + (int64_t)t * R;

    for (int r = (int)threadIdx.x; r < R; r += (int)blockDim.x) {
        // scale/q0 per (out_channel=r, group=g)
        int64_t idx = (int64_t)r * G + g;
        float s  = scales[idx];
        float q0 = qzeros[idx];

        float x = W_row[r];
        float err, deq;
        uint8_t q;
        quantize_scalar_u8(x, s, q0, maxq, err, q, deq);

        W_row[r] = deq;
        Q_row[r] = q;
        E_row[r] = err;
    }
}

// Launcher wrapper
static void launch_gptq_quantize_block_fp32(
    torch::Tensor W,
    torch::Tensor qweight,
    torch::Tensor Eblk,
    torch::Tensor scales,
    torch::Tensor qzeros,
    torch::Tensor g_idx,
    int64_t block_start,
    int64_t B,
    int64_t group_size,
    int64_t bits
) {
    CHECK_CUDA(W); CHECK_CUDA(qweight); CHECK_CUDA(Eblk); CHECK_CUDA(scales); CHECK_CUDA(qzeros);
    CHECK_CONTIGUOUS(W); CHECK_CONTIGUOUS(qweight); CHECK_CONTIGUOUS(Eblk); CHECK_CONTIGUOUS(scales); CHECK_CONTIGUOUS(qzeros);
    CHECK_FP32(W); CHECK_U8(qweight); CHECK_FP32(Eblk); CHECK_FP32(scales); CHECK_FP32(qzeros);
    if (g_idx.has_value()) {
        CHECK_CUDA(*g_idx);
        CHECK_CONTIGUOUS(*g_idx);
        CHECK_I32(*g_idx);
    }

    const int C = (int)W.size(0);
    const int R = (int)W.size(1);
    TORCH_CHECK(qweight.size(0) == C && qweight.size(1) == R, "qweight shape must match W [C,R]");

    TORCH_CHECK(Eblk.size(0) == B && Eblk.size(1) == R, "Eblk must be [B,R]");

    TORCH_CHECK(group_size > 0, "group_size must be > 0");
    const int G = (int)ceil_div_int64(C, group_size);
    TORCH_CHECK(scales.numel() == (int64_t)R * G, "scales must have numel == R*G (layout [R,G])");
    TORCH_CHECK(qzeros.numel() == (int64_t)R * G, "qzeros must have numel == R*G (layout [R,G])");

    const int threads = 256; // simple/default
    const dim3 block(threads);
    const dim3 grid((unsigned)B);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

    const int32_t* gptr = g_idx.data_ptr<int>();//g_idx.has_value() ? (const int32_t*)() : nullptr;

    gptq_quantize_block_kernel_fp32<<<grid, block, 0, stream>>>(
        W.data_ptr<float>(),
        qweight.data_ptr<uint8_t>(),
        Eblk.data_ptr<float>(),
        scales.data_ptr<float>(),
        qzeros.data_ptr<float>(),
        gptr,
        C,
        R,
        G,
        (int)block_start,
        (int)B,
        (int)group_size,
        (int)bits
    );
}

// -----------------------------
// 2) TRSM: (U_block^T) E^T = Delta^T, in-place on Eblk
// -----------------------------

// Loads U_block (upper) into shared, then performs forward substitution per column r.
// Eblk is [B,R]. After solve, Eblk becomes E.
template <int MAX_B>
__global__ void gptq_trsm_block_kernel_fp32(
    float* __restrict__ Eblk,       // [B,R] in/out
    const float* __restrict__ U,    // [C,C] upper-tri
    int C,
    int R,
    int block_start,
    int B
) {
    __shared__ float U_sh[MAX_B * MAX_B];
    __shared__ float inv_diag[MAX_B];

    // Load U_block into shared (only upper part needed).
    for (int idx = (int)threadIdx.x; idx < B * B; idx += (int)blockDim.x) {
        int i = idx / B;
        int j = idx - i * B;
        float v = 0.0f;
        if (j >= i) {
            v = U[(int64_t)(block_start + i) * C + (block_start + j)];
        }
        U_sh[i * MAX_B + j] = v;
    }
    // Precompute inv diagonal
    for (int i = (int)threadIdx.x; i < B; i += (int)blockDim.x) {
        float d = U_sh[i * MAX_B + i];
        inv_diag[i] = 1.0f / (d + 1e-20f);
    }
    __syncthreads();

    int r = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (r >= R) return;

    // Forward substitution on L = U^T.
    // L(i,k) = U(k,i). We use U_sh[k * MAX_B + i].
    for (int i = 0; i < B; ++i) {
        float acc = Eblk[(int64_t)i * R + r];
        #pragma unroll
        for (int k = 0; k < MAX_B; ++k) {
            if (k >= i) break;
            acc -= U_sh[k * MAX_B + i] * Eblk[(int64_t)k * R + r];
        }
        acc *= inv_diag[i];
        Eblk[(int64_t)i * R + r] = acc;
    }
}

static void launch_gptq_trsm_block_fp32(
    torch::Tensor Eblk,
    torch::Tensor U,
    int64_t block_start,
    int64_t B
) {
    CHECK_CUDA(Eblk); CHECK_CUDA(U);
    CHECK_CONTIGUOUS(Eblk); CHECK_CONTIGUOUS(U);
    CHECK_FP32(Eblk); CHECK_FP32(U);

    const int C = (int)U.size(0);
    TORCH_CHECK(U.dim() == 2 && U.size(0) == U.size(1), "U must be [C,C]");

    const int R = (int)Eblk.size(1);
    TORCH_CHECK(Eblk.size(0) == B, "Eblk first dim must equal B");

    // Keep MAX_B small-ish for shared memory.
    constexpr int MAX_B = 128;
    TORCH_CHECK(B <= MAX_B, "B must be <= ", MAX_B);

    const int threads = 256;
    const dim3 block(threads);
    const dim3 grid((unsigned)ceil_div_int64(R, threads));

    cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

    gptq_trsm_block_kernel_fp32<MAX_B><<<grid, block, 0, stream>>>(
        Eblk.data_ptr<float>(),
        U.data_ptr<float>(),
        C,
        R,
        (int)block_start,
        (int)B
    );
}

// -----------------------------
// 3) Full solver (block loop + tail update)
// -----------------------------

// Returns qweight [C,R] uint8. Mutates W in-place to dequantized values.
torch::Tensor gptq_solve_fp32(
    torch::Tensor W,            // [C,R] fp32
    torch::Tensor Hinv_u,       // [C,C] fp32 upper
    torch::Tensor scales,       // [R*G] fp32
    torch::Tensor qzeros,       // [R*G] fp32
    int64_t bits,
    int64_t group_size,
    int64_t block_size,
    bool symmetric,
    torch::Tensor g_idx
) {
    (void)symmetric; // currently unused in solver; quantize_scalar clamps q0 and uses it.

    CHECK_CUDA(W); CHECK_CUDA(Hinv_u); CHECK_CUDA(scales); CHECK_CUDA(qzeros);
    CHECK_CONTIGUOUS(W); CHECK_CONTIGUOUS(Hinv_u); CHECK_CONTIGUOUS(scales); CHECK_CONTIGUOUS(qzeros);
    CHECK_FP32(W); CHECK_FP32(Hinv_u); CHECK_FP32(scales); CHECK_FP32(qzeros);

    TORCH_CHECK(W.dim() == 2, "W must be 2D [C,R]");
    TORCH_CHECK(Hinv_u.dim() == 2 && Hinv_u.size(0) == Hinv_u.size(1), "Hinv_u must be [C,C]");

    const int64_t C = W.size(0);
    const int64_t R = W.size(1);
    TORCH_CHECK(Hinv_u.size(0) == C, "Hinv_u size mismatch with W");

    TORCH_CHECK(bits >= 2 && bits <= 8, "bits must be in [2,8]");
    TORCH_CHECK(group_size > 0, "group_size must be > 0");
    TORCH_CHECK(block_size > 0, "block_size must be > 0");

    const int64_t G = ceil_div_int64(C, group_size);
    TORCH_CHECK(scales.numel() == R * G, "scales must have numel == R*G");
    TORCH_CHECK(qzeros.numel() == R * G, "qzeros must have numel == R*G");

    

    // Allocate qweight [C,R]
    auto qweight = at::empty({C, R}, torch::TensorOptions().dtype(at::kByte).device(W.device()));

    // Temporary Eblk [B,R] (Delta then E)
    // We'll allocate at max block_size and slice per block if tail smaller.
    auto Ebuf = at::empty({block_size, R}, torch::TensorOptions().dtype(at::kFloat).device(W.device()));

    // Block loop
    for (int64_t block_start = 0; block_start < C; block_start += block_size) {
        int64_t block_end = std::min<int64_t>(block_start + block_size, C);
        int64_t B = block_end - block_start;

        // Slice Eblk = Ebuf[0:B]
        auto Eblk = Ebuf.narrow(0, 0, B).contiguous();

        // Quantize rows in block, fill Delta into Eblk
        launch_gptq_quantize_block_fp32(
            W,
            qweight,
            Eblk,
            scales,
            qzeros,
            g_idx,
            block_start,
            B,
            group_size,
            bits
        );

        // TRSM: Eblk = solve(U_block^T, Delta)
        launch_gptq_trsm_block_fp32(Eblk, Hinv_u, block_start, B);

        // Update tail W[block_end:,:] -= H_cross^T @ Eblk
        if (block_end < C) {
            auto H_cross = Hinv_u.narrow(0, block_start, B).narrow(1, block_end, C - block_end);
            // Ensure contiguous for addmm
            auto Ht = H_cross.transpose(0, 1).contiguous(); // [C_tail, B]
            auto Ect = Eblk.contiguous();                  // [B, R]
            auto W_tail = W.narrow(0, block_end, C - block_end);
            // W_tail = 1*W_tail + (-1) * (Ht @ Ect)
            W_tail.addmm_(Ht, Ect, /*beta=*/1.0, /*alpha=*/-1.0);
        }
    }

    return qweight;
}
