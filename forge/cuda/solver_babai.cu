// solver_babai.cu
//
// Butterfly-blocked Babai quantization using A = Chol(H)^T (upper-triangular).
// Matches your conventions:
//   - W: [C, R] mutated in-place to dequantized values
//   - qweight: [C, R] uint8 codes
//   - qmeta: per-(row, group) scale/qzero/flags (same packed format)
// Key structure (butterfly / message passing):
//   - Process blocks J from right-to-left.
//   - Inside a block: run Babai back-to-front (sequential in i, parallel in r).
//   - After block is quantized: push one message to the prefix via GEMM:
//         Y_prefix -= A_prefix,J @ Q_J
//
// NOTE: This implementation maintains Y = A @ W0 as a float32 buffer and updates it
//       to avoid requiring F = Chol(H^{-1}) and to stay faithful to Babai with A.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>

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

// --- Vectorized load/store helpers -----------------------------------------

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

template <>
__device__ __forceinline__ void load_4<float>(const float* src, float* dst) {
    float4 v = *reinterpret_cast<const float4*>(src);
    dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
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

template <>
__device__ __forceinline__ void store_4<float>(float* dst, const float* src) {
    float4 v;
    v.x = src[0]; v.y = src[1]; v.z = src[2]; v.w = src[3];
    *reinterpret_cast<float4*>(dst) = v;
}

// --- QMeta decoding ---------------------------------------------------------

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
        // symmetric: center zero point
        constexpr float HALF = 0.5f;
        qzero_u8 = static_cast<uint8_t>((maxq_g + 1.0f) * HALF);
    }
    qzero_f = static_cast<float>(qzero_u8);
}

// --- Quant primitives -------------------------------------------------------

__device__ __forceinline__ void quantize_scalar(
    float x, float inv_s, float s, float q0, int maxq_i,
    uint8_t& q_out, float& deq_out
) {
    float biased = x * inv_s + q0;
    int q = __float2int_rn(biased);

    if (q < 0) q = 0;
    else if (q > maxq_i) q = maxq_i;

    deq_out = __fmaf_rn(static_cast<float>(q), s, -q0 * s);
    q_out   = static_cast<uint8_t>(q);
}

__device__ __forceinline__ void quantize_process_4(
    const float* x_vals,
    float inv_s, float s, float q0, int maxq_i,
    uint32_t& q_packed, float* deq_vals
) {
    uint8_t q_b[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        quantize_scalar(x_vals[i], inv_s, s, q0, maxq_i, q_b[i], deq_vals[i]);
    }
    q_packed =  (uint32_t)q_b[0]
              | ((uint32_t)q_b[1] << 8)
              | ((uint32_t)q_b[2] << 16)
              | ((uint32_t)q_b[3] << 24);
}

// ----------------------------------------------------------------------------
// Kernel: Babai inside one block J=[block_start, block_start+B)
//
// - Maintains and updates Y[J,:] in-place for intra-block dependencies.
// - Writes dequantized Q into W rows and also into Q_block (float).
// - Writes Z codes into qweight.
//
// A is full [C,C] float, but we only use A[J,J] (upper-tri) here.
// ----------------------------------------------------------------------------

template <typename scalar_t, int MAX_B, bool VECTORIZED>
__global__ void babai_quant_block_kernel(
    scalar_t* __restrict__ W,              // [C, R] (write deq)
    uint8_t*  __restrict__ qweight,        // [C, R] (write codes)
    const QMetaPacked* __restrict__ qmeta, // [C*G]
    const float* __restrict__ A,           // [C, C] float (upper-tri)
    float* __restrict__ Y,                 // [C, R] float (updated in-place)
    float* __restrict__ Q_block,           // [MAX_B, R] float (write deq block)
    int C, int R, int G,
    int block_start, int B,
    int group_size,
    uint8_t bits
) {
    __shared__ float A_sh[MAX_B * MAX_B];

    const int tid = threadIdx.x;

    // Load A_block = A[block_start:block_start+B, block_start:block_start+B] into shared.
    for (int idx = tid; idx < B * B; idx += blockDim.x) {
        int r = idx / B;
        int c = idx - r * B;
        int gr = block_start + r;
        int gc = block_start + c;
        A_sh[r * MAX_B + c] = A[gr * C + gc];
    }
    __syncthreads();

    // Process local rows i = B-1..0 (back-to-front).
    if (VECTORIZED) {
        const int R_vec = R / 4;
        const int group_size_vec = group_size / 4;

        for (int i = B - 1; i >= 0; --i) {
            const int j = block_start + i;
            const float diag = A_sh[i * MAX_B + i];
            const float inv_diag = 1.0f / diag;

            auto* Q_row_vec = reinterpret_cast<uint32_t*>(qweight + j * R);
            auto* W_row     = W + j * R;
            float* Y_row    = Y + j * R;
            float* QB_row   = Q_block + i * R;

            // Quantize ω = Y[j,:] / A[j,j]
            for (int col_v = tid; col_v < R_vec; col_v += blockDim.x) {
                const int g = col_v / group_size_vec;
                const QMetaPacked m = qmeta[j * G + g];

                float s, inv_s, q0, maxq_g;
                decode_qmeta(qmeta_to_u32(m), s, inv_s, q0, maxq_g, bits);

                float omega[4];
                load_4(Y_row + col_v * 4, omega);
                #pragma unroll
                for (int t = 0; t < 4; ++t) omega[t] *= inv_diag;

                float deq[4];
                uint32_t q_packed;
                quantize_process_4(omega, inv_s, s, q0, static_cast<int>(maxq_g), q_packed, deq);

                // Store outputs
                Q_row_vec[col_v] = q_packed;
                store_4(W_row + col_v * 4, deq);   // write deq into W (cast)
                store_4(QB_row + col_v * 4, deq);  // keep float deq for prefix message

                // Intra-block update: for k = 0..i-1, Y[k,:] -= A[k,i] * Q_i
                #pragma unroll
                for (int k = 0; k < MAX_B; ++k) {
                    if (k >= i) break;
                    const float a_ki = A_sh[k * MAX_B + i];
                    float* Yk = Y + (block_start + k) * R + col_v * 4;

                    float yk[4];
                    load_4(Yk, yk);
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) yk[t] -= a_ki * deq[t];
                    store_4(Yk, yk);
                }
            }

            __syncthreads(); // ensure all columns updated before next i uses Y[i-1,:]
        }
    } else {
        for (int i = B - 1; i >= 0; --i) {
            const int j = block_start + i;
            const float diag = A_sh[i * MAX_B + i];
            const float inv_diag = 1.0f / diag;

            auto* W_row   = W + j * R;
            auto* Q_row   = qweight + j * R;
            float* Y_row  = Y + j * R;
            float* QB_row = Q_block + i * R;

            for (int col = tid; col < R; col += blockDim.x) {
                const int g = col / group_size;
                const QMetaPacked m = qmeta[j * G + g];

                float s, inv_s, q0, maxq_g;
                decode_qmeta(qmeta_to_u32(m), s, inv_s, q0, maxq_g, bits);

                float omega = Y_row[col] * inv_diag;

                uint8_t q_byte;
                float deq;
                quantize_scalar(omega, inv_s, s, q0, static_cast<int>(maxq_g), q_byte, deq);

                Q_row[col]   = q_byte;
                W_row[col]   = static_cast<scalar_t>(deq);
                QB_row[col]  = deq;

                // Intra-block update
                for (int k = 0; k < i; ++k) {
                    const float a_ki = A_sh[k * MAX_B + i];
                    Y[(block_start + k) * R + col] -= a_ki * deq;
                }
            }

            __syncthreads();
        }
    }
}

} // namespace

// ----------------------------------------------------------------------------
// Host wrapper
// ----------------------------------------------------------------------------

torch::Tensor babai_solver_cuda(
    torch::Tensor weight,        // [C, R]
    torch::Tensor A_chol_t,      // [C, C] = Chol(H)^T (upper-tri)
    torch::Tensor qmeta_bytes,   // [C, G, 4] or [C*G, 4]
    int64_t group_size,
    int64_t bits,
    int64_t block_size
) {
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(A_chol_t.is_cuda(), "A_chol_t must be CUDA");
    TORCH_CHECK(weight.dim() == 2, "weight must be [C, R]");
    TORCH_CHECK(A_chol_t.dim() == 2, "A_chol_t must be [C, C]");
    TORCH_CHECK(A_chol_t.size(0) == weight.size(0) && A_chol_t.size(1) == weight.size(0),
                "A_chol_t must match CxC");

    weight     = weight.contiguous();
    A_chol_t   = A_chol_t.contiguous();
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
    TORCH_CHECK(G > 0, "Invalid G derived from qmeta_bytes");

    // Clamp block size
    constexpr int MAX_BLOCK_SIZE = 32;
    if (block_size <= 0 || block_size > MAX_BLOCK_SIZE) block_size = MAX_BLOCK_SIZE;
    if (block_size > C) block_size = C;

    // Flatten qmeta
    auto qmeta_flat = (qmeta_bytes.dim() == 3) ? qmeta_bytes.view({C * G, 4}) : qmeta_bytes;

    // Output codes
    auto qweight = torch::empty({C, R}, torch::TensorOptions().dtype(torch::kUInt8).device(weight.device()));

    // Work buffers:
    //   A_f: float32 view of A
    //   Y:   float32 Babai residual state Y = A * W0
    auto A_f = A_chol_t.to(torch::kFloat).contiguous();
    auto W0f = weight.to(torch::kFloat); // uses current weight as "original"
    auto Y   = torch::matmul(A_f, W0f).contiguous(); // [C, R] float32

    // Float buffer for current block Q_J (dequant values), used for prefix message update
    auto Q_block = torch::empty({block_size, R}, torch::TensorOptions().dtype(torch::kFloat).device(weight.device()));

    auto stream = at::cuda::getCurrentCUDAStream();

    const bool use_vectorized = (R % 4 == 0) && (group_size % 4 == 0);
    constexpr int WARPS_PER_BLOCK = 8;
    const int threads = WARPS_PER_BLOCK * 32;

    // Process blocks from right to left (default Babai order)
    for (int64_t block_end = C; block_end > 0; block_end -= block_size) {
        int64_t block_start = std::max<int64_t>(0, block_end - block_size);
        int64_t B_long      = block_end - block_start;
        int     B           = static_cast<int>(B_long);

        // Quantize this block + intra-block updates on Y
        AT_DISPATCH_SWITCH(weight.scalar_type(), "babai_quant_block",
            AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
                if (use_vectorized) {
                    babai_quant_block_kernel<float, MAX_BLOCK_SIZE, true><<<1, threads, 0, stream>>>(
                        weight.data_ptr<float>(),
                        qweight.data_ptr<uint8_t>(),
                        reinterpret_cast<const QMetaPacked*>(qmeta_flat.data_ptr<uint8_t>()),
                        A_f.data_ptr<float>(),
                        Y.data_ptr<float>(),
                        Q_block.data_ptr<float>(),
                        (int)C, (int)R, (int)G,
                        (int)block_start, B,
                        (int)group_size,
                        (uint8_t)bits
                    );
                } else {
                    babai_quant_block_kernel<float, MAX_BLOCK_SIZE, false><<<1, threads, 0, stream>>>(
                        weight.data_ptr<float>(),
                        qweight.data_ptr<uint8_t>(),
                        reinterpret_cast<const QMetaPacked*>(qmeta_flat.data_ptr<uint8_t>()),
                        A_f.data_ptr<float>(),
                        Y.data_ptr<float>(),
                        Q_block.data_ptr<float>(),
                        (int)C, (int)R, (int)G,
                        (int)block_start, B,
                        (int)group_size,
                        (uint8_t)bits
                    );
                }
            })
            AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
                if (use_vectorized) {
                    babai_quant_block_kernel<at::Half, MAX_BLOCK_SIZE, true><<<1, threads, 0, stream>>>(
                        weight.data_ptr<at::Half>(),
                        qweight.data_ptr<uint8_t>(),
                        reinterpret_cast<const QMetaPacked*>(qmeta_flat.data_ptr<uint8_t>()),
                        A_f.data_ptr<float>(),
                        Y.data_ptr<float>(),
                        Q_block.data_ptr<float>(),
                        (int)C, (int)R, (int)G,
                        (int)block_start, B,
                        (int)group_size,
                        (uint8_t)bits
                    );
                } else {
                    babai_quant_block_kernel<at::Half, MAX_BLOCK_SIZE, false><<<1, threads, 0, stream>>>(
                        weight.data_ptr<at::Half>(),
                        qweight.data_ptr<uint8_t>(),
                        reinterpret_cast<const QMetaPacked*>(qmeta_flat.data_ptr<uint8_t>()),
                        A_f.data_ptr<float>(),
                        Y.data_ptr<float>(),
                        Q_block.data_ptr<float>(),
                        (int)C, (int)R, (int)G,
                        (int)block_start, B,
                        (int)group_size,
                        (uint8_t)bits
                    );
                }
            })
            AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
                if (use_vectorized) {
                    babai_quant_block_kernel<at::BFloat16, MAX_BLOCK_SIZE, true><<<1, threads, 0, stream>>>(
                        weight.data_ptr<at::BFloat16>(),
                        qweight.data_ptr<uint8_t>(),
                        reinterpret_cast<const QMetaPacked*>(qmeta_flat.data_ptr<uint8_t>()),
                        A_f.data_ptr<float>(),
                        Y.data_ptr<float>(),
                        Q_block.data_ptr<float>(),
                        (int)C, (int)R, (int)G,
                        (int)block_start, B,
                        (int)group_size,
                        (uint8_t)bits
                    );
                } else {
                    babai_quant_block_kernel<at::BFloat16, MAX_BLOCK_SIZE, false><<<1, threads, 0, stream>>>(
                        weight.data_ptr<at::BFloat16>(),
                        qweight.data_ptr<uint8_t>(),
                        reinterpret_cast<const QMetaPacked*>(qmeta_flat.data_ptr<uint8_t>()),
                        A_f.data_ptr<float>(),
                        Y.data_ptr<float>(),
                        Q_block.data_ptr<float>(),
                        (int)C, (int)R, (int)G,
                        (int)block_start, B,
                        (int)group_size,
                        (uint8_t)bits
                    );
                }
            })
            AT_DISPATCH_CASE(at::ScalarType::Float8_e4m3fn, [&] {
                if (use_vectorized) {
                    babai_quant_block_kernel<c10::Float8_e4m3fn, MAX_BLOCK_SIZE, true><<<1, threads, 0, stream>>>(
                        weight.data_ptr<c10::Float8_e4m3fn>(),
                        qweight.data_ptr<uint8_t>(),
                        reinterpret_cast<const QMetaPacked*>(qmeta_flat.data_ptr<uint8_t>()),
                        A_f.data_ptr<float>(),
                        Y.data_ptr<float>(),
                        Q_block.data_ptr<float>(),
                        (int)C, (int)R, (int)G,
                        (int)block_start, B,
                        (int)group_size,
                        (uint8_t)bits
                    );
                } else {
                    babai_quant_block_kernel<c10::Float8_e4m3fn, MAX_BLOCK_SIZE, false><<<1, threads, 0, stream>>>(
                        weight.data_ptr<c10::Float8_e4m3fn>(),
                        qweight.data_ptr<uint8_t>(),
                        reinterpret_cast<const QMetaPacked*>(qmeta_flat.data_ptr<uint8_t>()),
                        A_f.data_ptr<float>(),
                        Y.data_ptr<float>(),
                        Q_block.data_ptr<float>(),
                        (int)C, (int)R, (int)G,
                        (int)block_start, B,
                        (int)group_size,
                        (uint8_t)bits
                    );
                }
            })
        );
        CUDA_CHECK(cudaGetLastError());

        // Butterfly message: push block's quantized contribution to the prefix
        if (block_start > 0) {
            // Y_prefix -= A_prefixJ @ Q_J
            auto Y_prefix = Y.narrow(0, 0, block_start);
            auto A_prefixJ = A_f.narrow(0, 0, block_start)
                               .narrow(1, block_start, B_long)
                               .contiguous(); // [block_start, B]
            auto Q_J = Q_block.narrow(0, 0, B_long); // [B, R]

            // Y_prefix = 1.0 * Y_prefix + (-1.0) * (A_prefixJ @ Q_J)
            Y_prefix.addmm_(A_prefixJ, Q_J, -1.0f, 1.0f);
        }
    }

    return qweight;
}
