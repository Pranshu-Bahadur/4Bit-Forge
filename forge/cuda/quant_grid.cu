#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <limits>
#include <cmath>
#include <tuple>

#define CUDA_CHECK(expr) \
  do { \
    cudaError_t _err = (expr); \
    TORCH_CHECK(_err == cudaSuccess, "CUDA error: ", cudaGetErrorString(_err)); \
  } while (0)

// =============================================================================
// INTERNAL (Device/Kernel) - Keep hidden/static
// =============================================================================

namespace { // Anonymous namespace for kernels is fine/good

struct QMetaPacked {
    int16_t  log2_scale_fp;
    uint8_t  qzero;
    uint8_t  flags;
};

__constant__ float c_p[1024];

__device__ __forceinline__ int16_t encode_scale_q88(float s) {
    float log2s = log2f(fmaxf(s, 1e-20f));
    float fp    = log2s * 256.0f;
    fp = fminf(fmaxf(fp, -32768.0f), 32767.0f);
    return static_cast<int16_t>(lrintf(fp));
}

__device__ __forceinline__ float decode_scale_q88(int16_t q) {
    float fp   = static_cast<float>(q) / 256.0f;
    return exp2f(fp);
}

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void build_group_meta_vec4_kernel_typed(
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

    if constexpr (std::is_same<scalar_t, float>::value) {
        int vecs = static_cast<int>(group_size / 4);
        int tail = static_cast<int>(group_size % 4);
        const float4* x4 = reinterpret_cast<const float4*>(x + base);

        for (int v = tid; v < vecs; v += blockDim.x) {
            float4 val = x4[v];
            float v_min = fminf(fminf(val.x, val.y), fminf(val.z, val.w));
            float v_max = fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w));
            local_min = fminf(local_min, v_min);
            local_max = fmaxf(local_max, v_max);
        }
        if (tail > 0 && tid == 0) {
            int64_t start = base + static_cast<int64_t>(vecs) * 4;
            for (int i = 0; i < tail; ++i) {
                float v = static_cast<float>(x[start + i]);
                local_min = fminf(local_min, v);
                local_max = fmaxf(local_max, v);
            }
        }
    } else {
        for (int64_t i = tid; i < group_size; i += blockDim.x) {
            float v = static_cast<float>(x[base + i]);
            local_min = fminf(local_min, v);
            local_max = fmaxf(local_max, v);
        }
    }

    extern __shared__ float sdata[];
    float* smin = sdata;
    float* smax = sdata + blockDim.x;

    smin[tid] = local_min;
    smax[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smin[tid] = fminf(smin[tid], smin[tid + stride]);
            smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        float xmin = smin[0];
        float xmax = smax[0];
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
        m.log2_scale_fp = encode_scale_q88(s);
        float q0_clamped = fminf(fmaxf(q0, 0.0f), maxq);
        m.qzero          = static_cast<uint8_t>(lrintf(q0_clamped));
        m.flags          = symmetric ? 1 : 0;
        qmeta[g] = m;
    }
}

template <typename scalar_t, bool IS_L2_NORM>
__global__ void mse_scale_groups_kernel_typed(
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

    int tid  = threadIdx.x;
    int64_t base = static_cast<int64_t>(g) * group_size;

    QMetaPacked m = qmeta[g];
    float base_s  = decode_scale_q88(m.log2_scale_fp);
    float q0      = float(m.qzero);

    extern __shared__ float smem[];
    float* cached_x       = smem;
    float* shared_partial = smem + group_size;

    for (int64_t i = tid; i < group_size; i += blockDim.x) {
        cached_x[i] = static_cast<float>(x[base + i]);
    }
    __syncthreads();

    float best_loss = 1e30f;
    float best_s    = base_s;

    int lane     = tid & 31;
    int wid      = tid >> 5;
    int numWarps = (blockDim.x + 31) / 32;

    for (int64_t k = 0; k < P; ++k) {
        float shrink = c_p[k];
        float s = base_s * shrink;
        if (s <= 0.0f) continue;

        float local_loss = 0.0f;
        for (int64_t i = tid; i < group_size; i += blockDim.x) {
            float v = cached_x[i];
            float q = rintf(v / s + q0);
            q = fminf(fmaxf(q, 0.0f), maxq);
            float y    = (q - q0) * s;
            float diff = fabsf(y - v);
            if (IS_L2_NORM) {
                local_loss += diff * diff;
            } else {
                local_loss += powf(diff, norm);
            }
        }

        local_loss = warpReduceSum(local_loss);

        if (lane == 0 && wid < 32) {
            shared_partial[wid] = local_loss;
        }
        __syncthreads();

        if (wid == 0) {
            float val = (lane < numWarps) ? shared_partial[lane] : 0.0f;
            val = warpReduceSum(val);
            if (lane == 0 && val < best_loss) {
                best_loss = val;
                best_s    = s;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        m.log2_scale_fp = encode_scale_q88(best_s);
        qmeta[g] = m;
    }
}

// Helpers for host launcher
static inline int align_to_warp(int threads) {
    return (threads + 31) & ~31;
}

static inline size_t calc_smem_gptq(int group_size, int /*threads*/) {
    size_t bytes_x       = static_cast<size_t>(group_size) * sizeof(float);
    size_t bytes_partial = 32 * sizeof(float);
    return bytes_x + bytes_partial;
}

static inline int choose_threads_that_fit(int group_size, size_t smem_cap) {
    int target_threads = std::min(group_size, 1024);
    int threads = align_to_warp(target_threads);
    if (threads < 32) threads = 32;
    while (threads >= 32) {
        size_t smem = calc_smem_gptq(group_size, threads);
        if (smem <= smem_cap) return threads;
        threads -= 32;
    }
    return 32;
}

inline torch::Tensor ensure_contiguous_same_dtype(torch::Tensor x) {
    if (!x.is_contiguous()) x = x.contiguous();
    return x;
}

} // End Anonymous Namespace for kernels

// =============================================================================
// EXTERNAL (Host Wrappers) - MUST be in Global Scope (No namespace)
// =============================================================================

// IMPORTANT: Using QMetaPacked here requires it to be visible. 
// Since struct QMetaPacked is in anonymous namespace, we redefine/expose it 
// or simply cast to void*/uint8_t* internally to avoid linkage ambiguity.
// Safest: Treat it as opaque bytes in the signature.

std::tuple<torch::Tensor, torch::Tensor> build_group_meta_packed_cuda(
    torch::Tensor x_groups,
    int64_t bit_width,
    bool symmetric
) {
    TORCH_CHECK(x_groups.is_cuda(), "x_groups must be CUDA");
    TORCH_CHECK(x_groups.dim() == 2, "x_groups must be [G, group_size]");

    auto G          = x_groups.size(0);
    auto group_size = x_groups.size(1);

    TORCH_CHECK(group_size % 32 == 0, "group_size must be a multiple of 32");

    x_groups = ensure_contiguous_same_dtype(x_groups);
    auto device = x_groups.device();

    // 4 bytes per group
    auto qmeta_tensor = torch::empty(
        {G, 4}, 
        torch::TensorOptions().dtype(torch::kUInt8).device(device)
    );

    int threads = std::min<int64_t>(256, group_size);
    threads = align_to_warp(threads);
    if (threads < 32) threads = 32;
    int blocks = static_cast<int>(G);

    size_t shmem_bytes = 2 * static_cast<size_t>(threads) * sizeof(float);

    auto stream = at::cuda::getCurrentCUDAStream();

    const auto dtype = x_groups.scalar_type();
    // Use the struct definition from the anonymous namespace via casting
    using QMetaLocal = QMetaPacked; 
    auto* qmeta_ptr = reinterpret_cast<QMetaLocal*>(qmeta_tensor.data_ptr<uint8_t>());

    if (dtype == torch::kFloat8_e4m3fn || dtype == torch::kFloat8_e4m3fnuz) {
        auto x_f32 = x_groups.to(torch::kFloat32).contiguous();
        const float* x_ptr = x_f32.data_ptr<float>();
        build_group_meta_vec4_kernel_typed<float><<<blocks, threads, shmem_bytes, stream>>>(
            x_ptr, qmeta_ptr, G, group_size, static_cast<int>(bit_width), symmetric
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            torch::kHalf, torch::kBFloat16, dtype, "build_group_meta_packed_cuda",
            [&]() {
                using scalar_t_ = scalar_t;
                const scalar_t_* x_ptr = x_groups.data_ptr<scalar_t_>();
                build_group_meta_vec4_kernel_typed<scalar_t_><<<blocks, threads, shmem_bytes, stream>>>(
                    x_ptr, qmeta_ptr, G, group_size, static_cast<int>(bit_width), symmetric
                );
            }
        );
    }

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
    TORCH_CHECK(x_groups.is_cuda(), "x_groups must be CUDA");
    TORCH_CHECK(p.is_cuda(),        "p must be CUDA");
    TORCH_CHECK(qmeta_bytes.is_cuda(), "qmeta_bytes must be CUDA");

    auto G          = x_groups.size(0);
    auto group_size = x_groups.size(1);
    auto P          = p.size(0);

    TORCH_CHECK(group_size % 32 == 0, "group_size must be multiple of 32");
    TORCH_CHECK(P > 0 && P <= 1024, "P must be in (0, 1024]");

    x_groups = ensure_contiguous_same_dtype(x_groups);
    p        = p.contiguous();

    auto device = x_groups.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    CUDA_CHECK(cudaMemcpyToSymbol(
        c_p, p.data_ptr<float>(), static_cast<size_t>(P) * sizeof(float), 0, cudaMemcpyDeviceToDevice
    ));

    const cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
    size_t smem_cap = prop->sharedMemPerBlockOptin ? (size_t)prop->sharedMemPerBlockOptin : (size_t)prop->sharedMemPerBlock;

    int threads = choose_threads_that_fit(static_cast<int>(group_size), smem_cap);
    if (threads > prop->maxThreadsPerBlock) threads = prop->maxThreadsPerBlock;
    threads = align_to_warp(threads);

    int blocks = static_cast<int>(G);
    size_t smem_bytes = calc_smem_gptq(static_cast<int>(group_size), threads);

    // Opt-in for large SMEM if needed (A100)
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(mse_scale_groups_kernel_typed<float, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        cudaFuncSetAttribute(mse_scale_groups_kernel_typed<float, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        // Add other template instantiations if using half/bf16 inputs for MSE
    }

    float maxq_f = static_cast<float>(maxq);
    float norm_f = static_cast<float>(norm);

    using QMetaLocal = QMetaPacked;
    auto* qmeta_ptr = reinterpret_cast<QMetaLocal*>(qmeta_bytes.data_ptr<uint8_t>());
    const auto dtype = x_groups.scalar_type();

    if (dtype == torch::kFloat8_e4m3fn || dtype == torch::kFloat8_e4m3fnuz) {
        auto x_f32 = x_groups.to(torch::kFloat32).contiguous();
        const float* x_ptr = x_f32.data_ptr<float>();
        if (std::fabs(norm_f - 2.0f) < 1e-5f) {
            mse_scale_groups_kernel_typed<float, true><<<blocks, threads, smem_bytes, stream>>>(
                x_ptr, qmeta_ptr, G, group_size, P, maxq_f, norm_f);
        } else {
            mse_scale_groups_kernel_typed<float, false><<<blocks, threads, smem_bytes, stream>>>(
                x_ptr, qmeta_ptr, G, group_size, P, maxq_f, norm_f);
        }
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            torch::kHalf, torch::kBFloat16, dtype, "mse_scale_groups_packed_cuda",
            [&]() {
                using scalar_t_ = scalar_t;
                const scalar_t_* x_ptr = x_groups.data_ptr<scalar_t_>();
                if (std::fabs(norm_f - 2.0f) < 1e-5f) {
                    mse_scale_groups_kernel_typed<scalar_t_, true><<<blocks, threads, smem_bytes, stream>>>(
                        x_ptr, qmeta_ptr, G, group_size, P, maxq_f, norm_f);
                } else {
                    mse_scale_groups_kernel_typed<scalar_t_, false><<<blocks, threads, smem_bytes, stream>>>(
                        x_ptr, qmeta_ptr, G, group_size, P, maxq_f, norm_f);
                }
            }
        );
    }

    return qmeta_bytes;
}