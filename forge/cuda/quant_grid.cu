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
    __half  scale;
    uint8_t  qzero;
    uint8_t  flags;
};

// Constant memory for search grid (up to 1024 candidates)
__constant__ float c_p[1024];

// ---- PackBoost-Style Intrinsics ------------------------------------

__device__ __forceinline__ float fast_log2(float x) { return log2f(x); }
__device__ __forceinline__ float fast_exp2(float x) { return exp2f(x); }

// Single instruction rounding (Round to Nearest Even)
__device__ __forceinline__ float fast_round(float x) {
    return __float2int_rn(x); 
}



// ---- Butterfly Reductions ------------------------------------------

template <typename T>
__device__ __forceinline__ T butterflyReduceSum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T butterflyReduceMin(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T butterflyReduceMax(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Helper to convert arbitrary types (like uint8 FP8) to float for math
template <typename T>
__device__ __forceinline__ float val_to_float(T val) {
    return static_cast<float>(val);
}

// FP8 specialization (assuming stored as uint8_t)
template <>
__device__ __forceinline__ float val_to_float<uint8_t>(uint8_t val) {
    __nv_fp8_e4m3 fp8_val = *reinterpret_cast<__nv_fp8_e4m3*>(&val);
    return float(fp8_val);
}

inline torch::Tensor ensure_contiguous_same_dtype(torch::Tensor x) {
    if (!x.is_contiguous()) x = x.contiguous();
    return x;
}

static inline int align_to_warp(int threads) {
    return (threads + 31) & ~31;
}

// ====================================================================
// 1) META BUILDER (ABSMAX / RANGE)
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
    int g = blockIdx.x;
    if (g >= G) return;

    int tid  = threadIdx.x;
    int64_t base = static_cast<int64_t>(g) * group_size;

    float local_min = 1e30f;
    float local_max = -1e30f;

    // Vectorized Loads
    constexpr int bytes_per_load = 16;
    constexpr int elems_per_load = bytes_per_load / sizeof(scalar_t);
    const int4* x_vec = reinterpret_cast<const int4*>(x + base);
    int total_vecs = static_cast<int>(group_size / elems_per_load);

    for (int i = tid; i < total_vecs; i += blockDim.x) {
        int4 packed = x_vec[i];
        const scalar_t* vals = reinterpret_cast<const scalar_t*>(&packed);
        #pragma unroll
        for (int k = 0; k < elems_per_load; ++k) {
            float v = val_to_float(vals[k]);
            local_min = fminf(local_min, v);
            local_max = fmaxf(local_max, v);
        }
    }

    int tail_start = total_vecs * elems_per_load;
    for (int idx = tail_start + tid; idx < group_size; idx += blockDim.x) {
        float v = val_to_float(x[base + idx]);
        local_min = fminf(local_min, v);
        local_max = fmaxf(local_max, v);
    }

    local_min = butterflyReduceMin(local_min);
    local_max = butterflyReduceMax(local_max);

    if (tid == 0) {
        float xmin = local_min;
        float xmax = local_max;
        float maxq = float((1 << bit_width) - 1);
        float eps  = 1e-12f;
        float s, q0;

        if (symmetric) {
            float amax = fmaxf(fabsf(xmin), fabsf(xmax));
            s  = (2.0f / maxq) * amax + eps;
            q0 = 0.5f * (maxq + 1.0f);
        } else {
            s  = (xmax - xmin) / maxq + eps;
            float q = -xmin / s;
            q = fminf(fmaxf(q, 0.0f), maxq);
            q0 = rintf(q);
        }

        QMetaPacked m;
        m.scale = __float2half_rn(s);
        float q0_clamped = fminf(fmaxf(q0, 0.0f), maxq);
        m.qzero          = static_cast<uint8_t>(lrintf(q0_clamped));
        m.flags          = symmetric ? 1 : 0;
        qmeta[g] = m;
    }
}

// ====================================================================
// 2) MSE SCALE SEARCH 
// ====================================================================

template <typename scalar_t, bool IS_L2_NORM>
__global__ void mse_search_kernel_nosmem(
    const scalar_t* __restrict__ x,
    QMetaPacked* __restrict__ qmeta,
    int64_t G,
    int64_t group_size,
    int64_t P,
    float maxq,
    float norm
) {
    int g = blockIdx.x;
    if (g >= G) return;

    const int lane = threadIdx.x;    // 0..31
    const int64_t base = static_cast<int64_t>(g) * group_size;

    QMetaPacked m = qmeta[g];
    float base_s  = __half2float(m.scale);
    float q0      = static_cast<float>(m.qzero);

    float best_loss = FLT_MAX;
    float best_s    = base_s;

    int64_t k   = 0;
    int64_t P4  = P & ~3;   // largest multiple of 4 <= P

    // ---- Process candidates in chunks of 4 ----
    for (; k < P4; k += 4) {
        float p0 = c_p[k + 0];
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
        for (int64_t idx = lane; idx < group_size; idx += 32) {
            float v = val_to_float(x[base + idx]);

            // candidate 0
            float q = fast_round(fmaf(v, rcp0, q0));   // v * rcp0 + q0
            q       = fminf(fmaxf(q, 0.0f), maxq);
            float diff = fmaf(q - q0, s0, -v);         // (q - q0)*s0 - v
            if (IS_L2_NORM) {
                l0 = fmaf(diff, diff, l0);
            } else {
                float e = fmaxf(fabsf(diff), 1e-20f);
                float lg = __logf(e);
                float val = __expf(lg * norm);
                l0 += val;
            }

            // candidate 1
            q    = fast_round(fmaf(v, rcp1, q0));
            q    = fminf(fmaxf(q, 0.0f), maxq);
            diff = fmaf(q - q0, s1, -v);
            if (IS_L2_NORM) {
                l1 = fmaf(diff, diff, l1);
            } else {
                float e = fmaxf(fabsf(diff), 1e-20f);
                float lg = __logf(e);
                float val = __expf(lg * norm);
                l1 += val;
            }

            // candidate 2
            q    = fast_round(fmaf(v, rcp2, q0));
            q    = fminf(fmaxf(q, 0.0f), maxq);
            diff = fmaf(q - q0, s2, -v);
            if (IS_L2_NORM) {
                l2 = fmaf(diff, diff, l2);
            } else {
                float e = fmaxf(fabsf(diff), 1e-20f);
                float lg = __logf(e);
                float val = __expf(lg * norm);
                l2 += val;
            }

            // candidate 3
            q    = fast_round(fmaf(v, rcp3, q0));
            q    = fminf(fmaxf(q, 0.0f), maxq);
            diff = fmaf(q - q0, s3, -v);
            if (IS_L2_NORM) {
                l3 = fmaf(diff, diff, l3);
            } else {
                float e = fmaxf(fabsf(diff), 1e-20f);
                float lg = __logf(e);
                float val = __expf(lg * norm);
                l3 += val;
            }
        }

        l0 = butterflyReduceSum(l0);
        l1 = butterflyReduceSum(l1);
        l2 = butterflyReduceSum(l2);
        l3 = butterflyReduceSum(l3);

        if (lane == 0) {
            if (l0 < best_loss) { best_loss = l0; best_s = s0; }
            if (l1 < best_loss) { best_loss = l1; best_s = s1; }
            if (l2 < best_loss) { best_loss = l2; best_s = s2; }
            if (l3 < best_loss) { best_loss = l3; best_s = s3; }
        }
    }

    // ---- Tail: remaining P % 4 candidates ----
    for (; k < P; ++k) {
        float p  = c_p[k];
        float s  = base_s * p;
        float rcp = 1.0f / s;

        float loss = 0.0f;

        #pragma unroll 4
        for (int64_t idx = lane; idx < group_size; idx += 32) {
            float v = val_to_float(x[base + idx]);

            float q = fast_round(fmaf(v, rcp, q0));
            q       = fminf(fmaxf(q, 0.0f), maxq);
            float diff = fmaf(q - q0, s, -v);

            if (IS_L2_NORM) {
                loss = fmaf(diff, diff, loss);
            } else {
                float e = fmaxf(fabsf(diff), 1e-20f);
                float lg = __logf(e);
                float val = __expf(lg * norm);
                loss += val;
            }
        }

        loss = butterflyReduceSum(loss);
        if (lane == 0 && loss < best_loss) {
            best_loss = loss;
            best_s    = s;
        }
    }

    if (lane == 0) {
        m.scale = __float2half_rn(best_s);
        qmeta[g]        = m;
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
    //@TODO fix this kernel...have set safe threads=32
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

    //int threads = std::min<int64_t>(256, group_size);
    //threads = align_to_warp(threads);
    int threads = 32;
    if (threads < 32) threads = 32;
    const int blocks = static_cast<int>(G);

    // SMEM for Meta Builder (optional block reduce if threads > 32)
    const int warps_per_block = threads / 32;
    const size_t smem_bytes = 0;

    //const size_t smem_bytes = 2 * static_cast<size_t>(warps_per_block) * sizeof(float);

    auto stream = at::cuda::getCurrentCUDAStream();

    if (dtype == c10::ScalarType::Float8_e4m3fn || dtype == c10::ScalarType::Float8_e4m3fnuz) {
        // Pass uint8_t pointer directly
        const uint8_t* x_ptr = reinterpret_cast<uint8_t*>(x_groups.data_ptr());
        build_group_meta_optimized<uint8_t><<<blocks, threads, smem_bytes, stream>>>(
            x_ptr, qmeta_ptr, G, group_size, static_cast<int>(bit_width), symmetric
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            torch::kHalf, torch::kBFloat16, dtype, "build_group_meta_packed_cuda",
            [&]() {
                using scalar_t_ = scalar_t;
                const scalar_t_* x_ptr = x_groups.data_ptr<scalar_t_>();
                build_group_meta_optimized<scalar_t_><<<blocks, threads, smem_bytes, stream>>>(
                    x_ptr, qmeta_ptr, G, group_size, static_cast<int>(bit_width), symmetric
                );
            }
        );
    }

    CUDA_CHECK(cudaGetLastError());

    float maxq_val = float((1 << bit_width) - 1);
    auto maxq = torch::full({}, maxq_val, x_groups.options().dtype(torch::kFloat32));

    //auto maxq = torch::full({}, maxq_val, x_groups.options().dtype(torch::kFloat32));

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

    auto device = x_groups.device();
    auto dtype  = x_groups.scalar_type();
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
    const int threads = 32;   // one warp per group
    const size_t smem_bytes = 0;  // no shared memory

    float maxq_f = static_cast<float>(maxq);
    float norm_f = static_cast<float>(norm);
    bool is_l2   = (std::fabs(norm_f - 2.0f) < 1e-5f);

    if (dtype == c10::ScalarType::Float8_e4m3fn ||
        dtype == c10::ScalarType::Float8_e4m3fnuz) {

        const uint8_t* x_ptr = reinterpret_cast<const uint8_t*>(x_groups.data_ptr());

        if (is_l2) {
            mse_search_kernel_nosmem<uint8_t, true>
                <<<blocks, threads, smem_bytes, stream>>>(
                    x_ptr, qmeta_ptr, G, group_size, P, maxq_f, norm_f
                );
        } else {
            mse_search_kernel_nosmem<uint8_t, false>
                <<<blocks, threads, smem_bytes, stream>>>(
                    x_ptr, qmeta_ptr, G, group_size, P, maxq_f, norm_f
                );
        }
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            torch::kHalf, torch::kBFloat16, dtype, "mse_scale_groups_packed_cuda",
            [&]() {
                using scalar_t_ = scalar_t;
                const scalar_t_* x_ptr = x_groups.data_ptr<scalar_t_>();

                if (is_l2) {
                    mse_search_kernel_nosmem<scalar_t_, true>
                        <<<blocks, threads, smem_bytes, stream>>>(
                            x_ptr, qmeta_ptr, G, group_size, P, maxq_f, norm_f
                        );
                } else {
                    mse_search_kernel_nosmem<scalar_t_, false>
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