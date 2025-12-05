#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cmath>
#include <tuple>
#include <cfloat>

#define CUDA_CHECK(expr)                                      \
  do {                                                        \
    cudaError_t _err = (expr);                                \
    TORCH_CHECK(_err == cudaSuccess, "CUDA error: ",          \
                cudaGetErrorString(_err));                    \
  } while (0)

namespace {

struct QMetaPacked {
    int16_t  log2_scale_fp;  // Q8.8 of log2(scale)
    uint8_t  qzero;          // zero-point (for asymmetric)
    uint8_t  flags;          // bit 0: symmetric flag
};

// Constant memory for search grid (up to 1024 candidates)
__constant__ float c_p[1024];

// ---- Intrinsics ----------------------------------------------------

__device__ __forceinline__ float fast_log2(float x) {
    return __log2f(x);
}
__device__ __forceinline__ float fast_exp2(float x) {
    return __exp2f(x);
}

// round-to-nearest-even, but returned as float
__device__ __forceinline__ float fast_round(float x) {
    return static_cast<float>(__float2int_rn(x));
}

__device__ __forceinline__ int16_t encode_scale_q88(float s) {
    float log2s = fast_log2(fmaxf(s, 1e-20f));
    float fp    = log2s * 256.0f;
    fp = fminf(fmaxf(fp, -32768.0f), 32767.0f);
    return static_cast<int16_t>(lrintf(fp));
}

__device__ __forceinline__ float decode_scale_q88(int16_t q) {
    float fp = static_cast<float>(q) * (1.0f / 256.0f);
    return fast_exp2(fp);
}

// ---- Warp reductions -----------------------------------------------

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T warpReduceMin(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T warpReduceMax(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// ---- Type → float helper -------------------------------------------

template <typename T>
__device__ __forceinline__ float val_to_float(T v) {
    return static_cast<float>(v);
}

// FP8 specialization (underlying byte → __nv_fp8_e4m3 → float)
template <>
__device__ __forceinline__ float val_to_float<uint8_t>(uint8_t v) {
    __nv_fp8_e4m3 fp8 = *reinterpret_cast<__nv_fp8_e4m3*>(&v);
    return static_cast<float>(fp8);
}

inline torch::Tensor ensure_contiguous_same_dtype(torch::Tensor x) {
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    return x;
}

// ====================================================================
// 1) META BUILDER (ABSMAX / RANGE)
//     - One warp per group, strided over group_size.
//     - Symmetric scale: amax / (2^{bits-1} - 1)
// ====================================================================

template <typename scalar_t>
__global__ void build_group_meta_optimized(
    const scalar_t* __restrict__ x,
    QMetaPacked* __restrict__ qmeta,
    int64_t G,
    int64_t group_size,
    int bit_width,
    bool symmetric
) {
    const int g = blockIdx.x;
    if (g >= G) return;

    const int lane = threadIdx.x;  // assume blockDim.x == 32
    const int64_t base = static_cast<int64_t>(g) * group_size;

    float local_min = 1e30f;
    float local_max = -1e30f;

    // Strided load over the group
    for (int64_t i = lane; i < group_size; i += 32) {
        float v = val_to_float(x[base + i]);
        local_min = fminf(local_min, v);
        local_max = fmaxf(local_max, v);
    }

    // Warp-wide reduction
    local_min = warpReduceMin(local_min);
    local_max = warpReduceMax(local_max);

    if (lane == 0) {
        float xmin = local_min;
        float xmax = local_max;
        float eps  = 1e-12f;

        QMetaPacked m;
        float s = 1.0f;
        float q0 = 0.0f;

        if (symmetric) {
            // Standard symmetric weight-only quant:
            //   q in [-max_int, max_int]  (max_int = 2^{b-1} - 1)
            float amax    = fmaxf(fabsf(xmin), fabsf(xmax));
            float max_int = float((1 << (bit_width - 1)) - 1);  // e.g. 7 for 4-bit
            s  = amax / (max_int + eps);
            q0 = 0.0f;  // symmetric zero-point
        } else {
            // Asymmetric per-group range mapping:
            //   x in [xmin, xmax] → q in [0, maxq]
            float maxq = float((1 << bit_width) - 1);
            s  = (xmax - xmin) / (maxq + eps);
            float q = -xmin / (s + eps);           // ideal zero-point
            q  = fminf(fmaxf(q, 0.0f), maxq);
            q0 = rintf(q);
        }

        // Encode
        m.log2_scale_fp = encode_scale_q88(s);
        m.qzero         = static_cast<uint8_t>(fminf(fmaxf(q0, 0.0f), 255.0f));
        m.flags         = symmetric ? 1 : 0;

        qmeta[g] = m;
    }
}

// ====================================================================
// 2) MSE SCALE SEARCH (Brute-force over grid c_p)
//     - Single warp per group
//     - Symmetric: q = round(v / s), clamp to [-max_int, max_int]
//     - Asymmetric: q = round(v / s + q0), clamp to [0, maxq]
// ====================================================================

template <typename scalar_t, bool IS_L2_NORM>
__global__ void mse_search_kernel(
    const scalar_t* __restrict__ x,
    QMetaPacked* __restrict__ qmeta,
    int64_t G,
    int64_t group_size,
    int64_t P,
    float maxq,
    float norm
) {
    const int g = blockIdx.x;
    if (g >= G) return;

    const int lane  = threadIdx.x;   // assume 32
    const int64_t base = static_cast<int64_t>(g) * group_size;

    QMetaPacked m = qmeta[g];
    float base_s  = decode_scale_q88(m.log2_scale_fp);
    float q0      = float(m.qzero);
    bool is_sym   = (m.flags & 1) != 0;

    extern __shared__ float cached_x[];

    // Cooperative load into shared memory as float32
    for (int64_t i = lane; i < group_size; i += 32) {
        cached_x[i] = val_to_float(x[base + i]);
    }
    __syncthreads();

    float best_loss = FLT_MAX;
    float best_s    = base_s;

    // Precompute symmetric clamp bound if needed
    float max_int = 0.0f;
    if (is_sym) {
        // max_int = 2^{b-1} - 1 where maxq = 2^{b} - 1
        max_int = 0.5f * (maxq - 1.0f);  // e.g. (15 - 1)/2 = 7 for 4-bit
    }

    // Unrolled 4-at-a-time loop over grid candidates
    int64_t k     = 0;
    int64_t P_vec = P & ~3;

    for (; k < P_vec; k += 4) {
        float p0 = c_p[k];
        float p1 = c_p[k + 1];
        float p2 = c_p[k + 2];
        float p3 = c_p[k + 3];

        float s0 = base_s * p0; float rcp0 = 1.0f / s0;
        float s1 = base_s * p1; float rcp1 = 1.0f / s1;
        float s2 = base_s * p2; float rcp2 = 1.0f / s2;
        float s3 = base_s * p3; float rcp3 = 1.0f / s3;

        float l0 = 0.0f;
        float l1 = 0.0f;
        float l2 = 0.0f;
        float l3 = 0.0f;

        #pragma unroll 4
        for (int64_t i = lane; i < group_size; i += 32) {
            float v = cached_x[i];

            auto accum_err = [&](float rcp, float s) -> float {
                float q;
                if (is_sym) {
                    // Symmetric: q in [-max_int, max_int]
                    q = fast_round(v * rcp);
                    q = fminf(fmaxf(q, -max_int), max_int);
                    float d = q * s - v;
                    return IS_L2_NORM ? d * d : powf(fabsf(d), norm);
                } else {
                    // Asymmetric: q in [0, maxq] with zero-point q0
                    q = fast_round(v * rcp + q0);
                    q = fminf(fmaxf(q, 0.0f), maxq);
                    float d = (q - q0) * s - v;
                    return IS_L2_NORM ? d * d : powf(fabsf(d), norm);
                }
            };

            l0 += accum_err(rcp0, s0);
            l1 += accum_err(rcp1, s1);
            l2 += accum_err(rcp2, s2);
            l3 += accum_err(rcp3, s3);
        }

        l0 = warpReduceSum(l0);
        l1 = warpReduceSum(l1);
        l2 = warpReduceSum(l2);
        l3 = warpReduceSum(l3);

        if (lane == 0) {
            if (l0 < best_loss) { best_loss = l0; best_s = s0; }
            if (l1 < best_loss) { best_loss = l1; best_s = s1; }
            if (l2 < best_loss) { best_loss = l2; best_s = s2; }
            if (l3 < best_loss) { best_loss = l3; best_s = s3; }
        }
    }

    // Tail
    for (; k < P; ++k) {
        float p   = c_p[k];
        float s   = base_s * p;
        float rcp = 1.0f / s;
        float loss = 0.0f;

        #pragma unroll 4
        for (int64_t i = lane; i < group_size; i += 32) {
            float v = cached_x[i];

            float q;
            if (is_sym) {
                q = fast_round(v * rcp);
                q = fminf(fmaxf(q, -max_int), max_int);
                float d = q * s - v;
                loss += (IS_L2_NORM ? d * d : powf(fabsf(d), norm));
            } else {
                q = fast_round(v * rcp + q0);
                q = fminf(fmaxf(q, 0.0f), maxq);
                float d = (q - q0) * s - v;
                loss += (IS_L2_NORM ? d * d : powf(fabsf(d), norm));
            }
        }

        loss = warpReduceSum(loss);
        if (lane == 0 && loss < best_loss) {
            best_loss = loss;
            best_s    = s;
        }
    }

    if (lane == 0) {
        m.log2_scale_fp = encode_scale_q88(best_s);
        qmeta[g] = m;
    }
}

} // anonymous namespace

// ======================================================================
// EXTERNAL HOST WRAPPERS
// ======================================================================

std::tuple<torch::Tensor, torch::Tensor> build_group_meta_packed_cuda(
    torch::Tensor x_groups,
    int64_t bit_width,
    bool symmetric
) {
    TORCH_CHECK(x_groups.is_cuda(), "x_groups must be CUDA");
    TORCH_CHECK(x_groups.dim() == 2, "x_groups must be [G, group_size]");

    x_groups = ensure_contiguous_same_dtype(x_groups);

    const auto G          = x_groups.size(0);
    const auto group_size = x_groups.size(1);
    TORCH_CHECK(group_size % 32 == 0, "group_size must be multiple of 32");

    auto device = x_groups.device();
    auto dtype  = x_groups.scalar_type();

    auto qmeta_tensor = torch::empty(
        {G, 4},
        torch::TensorOptions().dtype(torch::kUInt8).device(device)
    );
    using QMetaLocal = QMetaPacked;
    auto* qmeta_ptr = reinterpret_cast<QMetaLocal*>(qmeta_tensor.data_ptr<uint8_t>());

    const int blocks  = static_cast<int>(G);
    const int threads = 32;  // exactly one warp per group
    const size_t smem_bytes = 0;

    auto stream = at::cuda::getCurrentCUDAStream();

    if (dtype == c10::ScalarType::Float8_e4m3fn ||
        dtype == c10::ScalarType::Float8_e4m3fnuz) {

        const uint8_t* x_ptr = reinterpret_cast<uint8_t*>(x_groups.data_ptr());
        build_group_meta_optimized<uint8_t>
            <<<blocks, threads, smem_bytes, stream>>>(
                x_ptr, qmeta_ptr, G, group_size,
                static_cast<int>(bit_width), symmetric
            );
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            torch::kHalf, torch::kBFloat16, dtype,
            "build_group_meta_packed_cuda",
            [&]() {
                using scalar_t_ = scalar_t;
                const scalar_t_* x_ptr = x_groups.data_ptr<scalar_t_>();
                build_group_meta_optimized<scalar_t_>
                    <<<blocks, threads, smem_bytes, stream>>>(
                        x_ptr, qmeta_ptr, G, group_size,
                        static_cast<int>(bit_width), symmetric
                    );
            }
        );
    }

    CUDA_CHECK(cudaGetLastError());

    float maxq_val = float((1 << bit_width) - 1);
    auto maxq = torch::full({}, maxq_val, x_groups.options().dtype(torch::kFloat32));

    return std::make_tuple(qmeta_tensor, maxq);
}

torch::Tensor mse_scale_groups_packed_cuda(
    torch::Tensor x_groups,
    torch::Tensor p,
    torch::Tensor qmeta_bytes,
    double maxq,
    double norm
) {
    TORCH_CHECK(x_groups.is_cuda(),    "x_groups must be CUDA");
    TORCH_CHECK(p.is_cuda(),           "p must be CUDA");
    TORCH_CHECK(qmeta_bytes.is_cuda(), "qmeta_bytes must be CUDA");

    auto G          = x_groups.size(0);
    auto group_size = x_groups.size(1);
    auto P          = p.size(0);

    TORCH_CHECK(group_size % 32 == 0, "group_size must be multiple of 32");
    TORCH_CHECK(P > 0 && P <= 1024,   "P must be in (0, 1024]");

    x_groups = ensure_contiguous_same_dtype(x_groups);
    p        = p.contiguous();

    auto stream = at::cuda::getCurrentCUDAStream();

    CUDA_CHECK(cudaMemcpyToSymbol(
        c_p,
        p.data_ptr<float>(),
        static_cast<size_t>(P) * sizeof(float),
        0,
        cudaMemcpyDeviceToDevice
    ));

    using QMetaLocal = QMetaPacked;
    auto* qmeta_ptr = reinterpret_cast<QMetaLocal*>(qmeta_bytes.data_ptr<uint8_t>());

    const int blocks  = static_cast<int>(G);
    const int threads = 32;  // one warp per group
    const size_t smem_bytes = static_cast<size_t>(group_size) * sizeof(float);

    float maxq_f = static_cast<float>(maxq);
    float norm_f = static_cast<float>(norm);

    auto dtype = x_groups.scalar_type();

    if (dtype == c10::ScalarType::Float8_e4m3fn ||
        dtype == c10::ScalarType::Float8_e4m3fnuz) {

        const uint8_t* x_ptr = reinterpret_cast<uint8_t*>(x_groups.data_ptr());
        bool is_l2 = (std::fabs(norm_f - 2.0f) < 1e-5f);

        if (is_l2) {
            mse_search_kernel<uint8_t, true>
                <<<blocks, threads, smem_bytes, stream>>>(
                    x_ptr, qmeta_ptr, G, group_size, P, maxq_f, norm_f
                );
        } else {
            mse_search_kernel<uint8_t, false>
                <<<blocks, threads, smem_bytes, stream>>>(
                    x_ptr, qmeta_ptr, G, group_size, P, maxq_f, norm_f
                );
        }
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            torch::kHalf, torch::kBFloat16, dtype,
            "mse_scale_groups_packed_cuda",
            [&]() {
                using scalar_t_ = scalar_t;
                const scalar_t_* x_ptr = x_groups.data_ptr<scalar_t_>();
                bool is_l2 = (std::fabs(norm_f - 2.0f) < 1e-5f);

                if (is_l2) {
                    mse_search_kernel<scalar_t_, true>
                        <<<blocks, threads, smem_bytes, stream>>>(
                            x_ptr, qmeta_ptr, G, group_size, P, maxq_f, norm_f
                        );
                } else {
                    mse_search_kernel<scalar_t_, false>
                        <<<blocks, threads, smem_bytes, stream>>>(
                            x_ptr, qmeta_ptr, G, group_size, P, maxq_f, norm_f
                        );
                }
            }
        );
    }

    CUDA_CHECK(cudaGetLastError());
    return qmeta_bytes;
}
