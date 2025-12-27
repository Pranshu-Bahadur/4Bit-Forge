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

// -----------------------------------------------------------------------------
// Kernel 1: Babai quantize block (back-to-front) + intra-block propagation.
//
// IMPORTANT: qmeta layout is [R * G], where:
//   - R is output dim (columns of W_t)
//   - G = ceil(C / group_size) groups along input dim C
//
// W and qweight are transposed layout: [C, R].
// -----------------------------------------------------------------------------

template <typename scalar_t, int MAX_B>
__global__ void babai_quant_block_kernel_fast(
    scalar_t* __restrict__ W,                   // [C, R]
    uint8_t*  __restrict__ qweight,             // [C, R]
    const float* __restrict__ scales, //G*R
    const float* __restrict__ qzeros, ////G*R
    const float* __restrict__ A,                // [C, C] float (upper-tri expected)
    const float* __restrict__ invD_all,         // [C] float
    float* __restrict__ Eblk,                   // [B, R] float
    const int32_t* __restrict__ g_idx,         
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

    float x[MAX_B];

    for (int i = 0; i < MAX_B; ++i) x[i] = 0.0f;

    for (int i = 0; i < MAX_B; ++i) {
        if (i < B) {
            int row = block_start + i;
            x[i] = to_f32(W[row * R + r]);
        }
    }

    const int maxq_i = (1 << bits) - 1;

    // Back-to-front within this block
    for (int t = B - 1; t >= 0; --t) {
        int row = block_start + t;

        int g = g_idx ? (int)g_idx[row] : (row / group_size);
        if (g >= G) g = G - 1;

        float s = scales[G * r + g];
        float inv_s = 1/s;
        float q0 = qzeros[G * r + g];
        //q0 = nearbyintf(q0);
        //q0 = fminf(fmaxf(q0, 0.f), maxq_i);

        float err, deq;
        uint8_t qb;
        quantize_scalar(x[t], inv_s, s, q0, maxq_i, err, qb, deq);

        qweight[row * R + r] = qb;
        W[row * R + r]       = deq;
        Eblk[t * R + r]      = err;

        // Propagate to earlier rows i < t
        #pragma unroll
        for (int i = 0; i < t; ++i) {
                float alpha = S_sh[i * MAX_B + t];
                x[i] = __fmaf_rn(alpha, err, x[i]);
        }
    }
}

} // namespace

// -----------------------------------------------------------------------------
// Host wrapper
// -----------------------------------------------------------------------------
torch::Tensor babai_solver_cuda(
    torch::Tensor weight,      // [C, R]
    torch::Tensor A,           // [C, C] upper-tri = chol(H)^T
    torch::Tensor scales, // [R*G]
    torch::Tensor qzeros, // [R*G]
    int64_t group_size,
    int64_t bits,
    int64_t block_size,
    torch::Tensor g_idx,
    int G
) {
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(A.is_cuda(),      "A must be CUDA");
    TORCH_CHECK(scales.is_cuda(), "scales must be CUDA");
    TORCH_CHECK(qzeros.is_cuda(), "qzeros must be CUDA");


    TORCH_CHECK(weight.dim() == 2, "weight must be [C, R]");
    const int64_t C = weight.size(0);
    const int64_t R = weight.size(1);

    TORCH_CHECK(A.dim() == 2 && A.size(0) == C && A.size(1) == C, "A must be [C, C]");

    scales = scales.contiguous();
    qzeros = qzeros.contiguous();
    g_idx = g_idx.contiguous();
    // Determine G from qmeta tensor shape
    

    // Effective group_size along C and assert matches qmeta G
    TORCH_CHECK(group_size > 0, "group_size must be > 0");
    if (group_size > C) group_size = C;
    TORCH_CHECK(group_size % 32 == 0, "group_size must be multiple of 32 (effective group_size)");

    const int64_t expected_G = (C + group_size - 1) / group_size;
    TORCH_CHECK(expected_G == G,
                "qmeta G mismatch: got G=", G,
                " expected ceil(C/group_size)=", expected_G,
                " with C=", C, " group_size=", group_size);

    constexpr int MAX_B = 32;
    if (block_size <= 0 || block_size > MAX_B) block_size = MAX_B;
    if (block_size > C) block_size = C;

    auto qweight = torch::zeros({C, R},
        torch::TensorOptions().dtype(torch::kUInt8).device(weight.device()));

    auto Eblk = torch::zeros({block_size, R},
        torch::TensorOptions().dtype(at::kFloat).device(weight.device()));


    // Cast A and weight to float once
    auto A_f = (A.scalar_type() == at::ScalarType::Float) ? A : A.to(torch::kFloat);
    A_f = A_f.contiguous();

    if (weight.scalar_type() != at::ScalarType::Float) {
        weight = weight.to(torch::kFloat);
    }
    weight = weight.contiguous();

    auto invD_all = A_f.diagonal(0, 0, 1).reciprocal().contiguous();

    auto stream = at::cuda::getCurrentCUDAStream();

    auto A_tmp = torch::zeros({C, block_size},
        torch::TensorOptions().dtype(at::kFloat).device(weight.device()));

    constexpr int THREADS_Q = 128;

    for (int64_t block_end = C; block_end > 0; block_end -= block_size) {
        const int64_t block_start = std::max<int64_t>(0, block_end - block_size);
        const int64_t B_long      = block_end - block_start;
        const int     B           = static_cast<int>(B_long);

        auto Eblk_view = Eblk.narrow(0, 0, B_long); // [B, R]

        const int grid_q = (static_cast<int>(R) + THREADS_Q - 1) / THREADS_Q;

        babai_quant_block_kernel_fast<float, MAX_B><<<grid_q, THREADS_Q, 0, stream>>>(
            weight.data_ptr<float>(),
            qweight.data_ptr<uint8_t>(),
            scales.data_ptr<float>(),
            qzeros.data_ptr<float>(),
            A_f.data_ptr<float>(),
            invD_all.data_ptr<float>(),
            Eblk_view.data_ptr<float>(),
            g_idx.data_ptr<int32_t>(),
            (int)C, (int)R, (int)G,
            (int)block_start, B,
            (int)group_size,
            (uint8_t)bits
        );
        CUDA_CHECK(cudaGetLastError());

        if (block_start > 0) {
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
            CUDA_CHECK(cudaGetLastError());

            auto W_left = weight.narrow(0, 0, block_start);
            auto A_view = A_tmp.narrow(0, 0, block_start).narrow(1, 0, B_long);
            W_left.addmm_(A_view, Eblk_view, /*beta=*/1.0, /*alpha=*/1.0);
        }
    }

    // Return qweight in transposed solver layout: [C, R]. Caller can transpose if needed.
    return qweight.contiguous();
}