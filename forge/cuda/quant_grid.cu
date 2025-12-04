#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>


#include <cuda_fp8.h>   // for __nv_fp8_e4m3
#include <cuda_bf16.h>  // for __nv_bfloat16 if you want bf16
#include <cuda_fp16.h>  // for __half


#include <cstdint>
#include <limits>
#include <cmath>
#include <tuple>


#define CUDA_CHECK(expr) \
  do { \
    cudaError_t _err = (expr); \
    TORCH_CHECK(_err == cudaSuccess, "CUDA error: ", cudaGetErrorString(_err)); \
  } while (0)

namespace quant_grid {

// -----------------------------------------------------------------------------
// Metadata packing: QMetaPacked (4 bytes per group)
// -----------------------------------------------------------------------------

struct QMetaPacked {
    int16_t  log2_scale_fp;  // log2(scale) * 256 (Q8.8)
    uint8_t  qzero;          // zero-point in [0, maxq]
    uint8_t  flags;          // bit 0: symmetric
};

__device__ __forceinline__ int16_t encode_scale_q88(float s) {
    float log2s = log2f(fmaxf(s, 1e-20f));
    float fp    = log2s * 256.0f;
    fp = fminf(fmaxf(fp, -32768.0f), 32767.0f);
    return static_cast<int16_t>(lrintf(fp));
}

__device__ __forceinline__ float decode_scale_q88(int16_t q) {
    float fp   = static_cast<float>(q) / 256.0f;
    float s    = exp2f(fp);
    return s;
}

// -----------------------------------------------------------------------------
// Constant memory for shrink factors p[k]
// -----------------------------------------------------------------------------

__constant__ float c_p[1024];  // up to 1024 grid points

// -----------------------------------------------------------------------------
// Warp reduction helper
// -----------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    // Assumes full warp participation
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// -----------------------------------------------------------------------------
// Launch helpers
// -----------------------------------------------------------------------------

static inline int align_to_warp(int threads) {
    return (threads + 31) & ~31;
}

// Shared memory usage for MSE kernel:
//  - cached_x: group_size * float
//  - partials: 32 * float (one per warp, max 32 warps)
static inline size_t calc_smem_gptq(int group_size, int /*threads_per_block*/) {
    size_t bytes_x       = static_cast<size_t>(group_size) * sizeof(float);
    size_t bytes_partial = 32 * sizeof(float);
    return bytes_x + bytes_partial;
}

// Pick largest thread block that fits SMEM, respecting warp alignment
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

// Ensure tensor is contiguous but do not change dtype
inline torch::Tensor ensure_contiguous_same_dtype(torch::Tensor x) {
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    return x;
}

// -----------------------------------------------------------------------------
// Kernel 1: Build per-group min/max → initial scale & zero-point
// -----------------------------------------------------------------------------

template <typename scalar_t>
__global__ void build_group_meta_vec4_kernel_typed(
    const scalar_t* __restrict__ x,  // [G * group_size]
    QMetaPacked* __restrict__ qmeta, // [G]
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

    // Vectorized path for float
    if constexpr (std::is_same<scalar_t, float>::value) {
        int vecs = static_cast<int>(group_size / 4);
        int tail = static_cast<int>(group_size % 4);

        const float4* x4 = reinterpret_cast<const float4*>(x + base);

        for (int v = tid; v < vecs; v += blockDim.x) {
            float4 val = x4[v];
            float v_min = fminf(fminf(val.x, val.y), fminf(val.z, val.w));
            float v_max = fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w));
            if (v_min < local_min) local_min = v_min;
            if (v_max > local_max) local_max = v_max;
        }

        if (tail > 0 && tid == 0) {
            int64_t start = base + static_cast<int64_t>(vecs) * 4;
            for (int i = 0; i < tail; ++i) {
                float v = static_cast<float>(x[start + i]);
                if (v < local_min) local_min = v;
                if (v > local_max) local_max = v;
            }
        }
    } else {
        // Generic scalar path (half, bfloat16, etc.)
        for (int64_t i = tid; i < group_size; i += blockDim.x) {
            float v = static_cast<float>(x[base + i]);
            if (v < local_min) local_min = v;
            if (v > local_max) local_max = v;
        }
    }

    extern __shared__ float sdata[];
    float* smin = sdata;
    float* smax = sdata + blockDim.x;

    smin[tid] = local_min;
    smax[tid] = local_max;
    __syncthreads();

    // Block reduction for min/max
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = smin[tid + stride];
            if (a < smin[tid]) smin[tid] = a;
            float b = smax[tid + stride];
            if (b > smax[tid]) smax[tid] = b;
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

// -----------------------------------------------------------------------------
// Kernel 2: MSE-based scale refinement per group
// -----------------------------------------------------------------------------

template <typename scalar_t, bool IS_L2_NORM>
__global__ void mse_scale_groups_kernel_typed(
    const scalar_t* __restrict__ x,   // [G * group_size]
    QMetaPacked* __restrict__ qmeta,  // [G]
    int64_t G,
    int64_t group_size,
    int64_t P,
    float maxq,
    float norm
) {
    int g = blockIdx.x;  // one block per group
    if (g >= G) return;

    int tid  = threadIdx.x;
    int64_t base = static_cast<int64_t>(g) * group_size;

    QMetaPacked m = qmeta[g];
    float base_s  = decode_scale_q88(m.log2_scale_fp);
    float q0      = float(m.qzero);

    extern __shared__ float smem[];
    float* cached_x       = smem;                   // [group_size]
    float* shared_partial = smem + group_size;      // [32]

    // Cache group in shared memory (widen to float)
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
            float q = v / s + q0;
            q = rintf(q);
            if (q < 0.0f) q = 0.0f;
            if (q > maxq) q = maxq;
            float y    = (q - q0) * s;
            float diff = fabsf(y - v);

            if (IS_L2_NORM) {
                local_loss += diff * diff;
            } else {
                local_loss += powf(diff, norm);
            }
        }

        local_loss = warpReduceSum(local_loss);

        if (lane == 0 && wid < numWarps && wid < 32) {
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

// -----------------------------------------------------------------------------
// C++ entry: build_group_meta_packed_cuda
// -----------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> build_group_meta_packed_cuda(
    torch::Tensor x_groups,   // [G, group_size], any of {f32,f16,bf16,fp8_e4m3*}
    int64_t bit_width,
    bool symmetric
) {
    TORCH_CHECK(x_groups.is_cuda(), "x_groups must be CUDA");
    TORCH_CHECK(x_groups.dim() == 2, "x_groups must be [G, group_size]");

    auto G          = x_groups.size(0);
    auto group_size = x_groups.size(1);

    TORCH_CHECK(group_size % 32 == 0,
                "group_size must be a multiple of 32");

    x_groups = ensure_contiguous_same_dtype(x_groups);
    auto device = x_groups.device();

    auto qmeta_tensor = torch::empty(
        {G, static_cast<int64_t>(sizeof(QMetaPacked))},
        torch::TensorOptions().dtype(torch::kUInt8).device(device)
    );

    int threads = std::min<int64_t>(256, group_size);
    threads = align_to_warp(threads);
    if (threads < 32) threads = 32;
    int blocks = static_cast<int>(G);

    size_t shmem_bytes = 2 * static_cast<size_t>(threads) * sizeof(float); // smin + smax

    auto stream = c10::cuda::getCurrentCUDAStream();
    c10::cuda::CUDAGuard guard(device);

    const auto dtype = x_groups.scalar_type();

    auto* qmeta_ptr = reinterpret_cast<QMetaPacked*>(qmeta_tensor.data_ptr<uint8_t>());

    // For fp8 we go via float32 for now
    if (dtype == torch::kFloat8_e4m3fn || dtype == torch::kFloat8_e4m3fnuz) {
        auto x_f32 = x_groups.to(torch::kFloat32).contiguous();
        const float* x_ptr = x_f32.data_ptr<float>();

        build_group_meta_vec4_kernel_typed<float><<<blocks, threads, shmem_bytes, stream>>>(
            x_ptr, qmeta_ptr, G, group_size,
            static_cast<int>(bit_width),
            symmetric
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            torch::kHalf,
            torch::kBFloat16,
            dtype,
            "build_group_meta_packed_cuda",
            [&]() {
                using scalar_t_ = scalar_t;
                const scalar_t_* x_ptr = x_groups.data_ptr<scalar_t_>();

                build_group_meta_vec4_kernel_typed<scalar_t_><<<blocks, threads, shmem_bytes, stream>>>(
                    x_ptr, qmeta_ptr, G, group_size,
                    static_cast<int>(bit_width),
                    symmetric
                );
            }
        );
    }

    float maxq_val = float((1 << bit_width) - 1);
    auto maxq = torch::full(
        {},
        maxq_val,
        x_groups.options().dtype(torch::kFloat32)
    );

    return std::make_tuple(qmeta_tensor, maxq);
}

// -----------------------------------------------------------------------------
// C++ entry: mse_scale_groups_packed_cuda
// -----------------------------------------------------------------------------

torch::Tensor mse_scale_groups_packed_cuda(
    torch::Tensor x_groups,    // [G, group_size], same dtype set as above
    torch::Tensor p,           // [P], float32 shrink factors
    torch::Tensor qmeta_bytes, // [G, sizeof(QMetaPacked)], uint8
    double maxq,
    double norm
) {
    TORCH_CHECK(x_groups.is_cuda(), "x_groups must be CUDA");
    TORCH_CHECK(p.is_cuda(),        "p must be CUDA");
    TORCH_CHECK(qmeta_bytes.is_cuda(), "qmeta_bytes must be CUDA");

    TORCH_CHECK(x_groups.dim() == 2, "x_groups must be [G, group_size]");
    TORCH_CHECK(p.dim() == 1, "p must be 1D [P]");
    TORCH_CHECK(p.scalar_type() == torch::kFloat32, "p must be float32");

    auto G          = x_groups.size(0);
    auto group_size = x_groups.size(1);
    auto P          = p.size(0);

    TORCH_CHECK(group_size % 32 == 0, "group_size must be multiple of 32");
    TORCH_CHECK(P > 0 && P <= 1024, "P must be in (0, 1024]");

    TORCH_CHECK(qmeta_bytes.dim() == 2, "qmeta_bytes must be [G, 4]");
    TORCH_CHECK(qmeta_bytes.size(0) == G, "qmeta_bytes G mismatch");
    TORCH_CHECK(qmeta_bytes.size(1) == (int64_t)sizeof(QMetaPacked),
        "qmeta_bytes second dim must equal sizeof(QMetaPacked)");

    x_groups = ensure_contiguous_same_dtype(x_groups);
    p        = p.contiguous();

    auto device = x_groups.device();
    c10::cuda::CUDAGuard guard(device);
    auto stream = c10::cuda::getCurrentCUDAStream();

    // Copy shrink factors to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(
        c_p,
        p.data_ptr<float>(),
        static_cast<size_t>(P) * sizeof(float),
        0,
        cudaMemcpyDeviceToDevice
    ));

    const cudaDeviceProp* prop = c10::cuda::getCurrentDeviceProperties();
    size_t smem_cap = prop->sharedMemPerBlockOptin
                      ? (size_t)prop->sharedMemPerBlockOptin
                      : (size_t)prop->sharedMemPerBlock;

    int threads = choose_threads_that_fit(static_cast<int>(group_size), smem_cap);
    if (threads > prop->maxThreadsPerBlock) {
        threads = align_to_warp(prop->maxThreadsPerBlock);
        if (threads < 32) threads = 32;
    }

    int blocks = static_cast<int>(G);
    size_t smem_bytes = calc_smem_gptq(static_cast<int>(group_size), threads);

    TORCH_CHECK(smem_bytes <= smem_cap,
        "Requested shared memory exceeds per-block limit");

    float maxq_f = static_cast<float>(maxq);
    float norm_f = static_cast<float>(norm);

    auto* qmeta_ptr = reinterpret_cast<QMetaPacked*>(qmeta_bytes.data_ptr<uint8_t>());
    const auto dtype = x_groups.scalar_type();

    // We don't expect smem_bytes to exceed 48KB for typical group sizes.
    // If you ever push group_size very high, you can add cudaFuncSetAttribute
    // here for the specific kernel instantiations.

    if (dtype == torch::kFloat8_e4m3fn || dtype == torch::kFloat8_e4m3fnuz) {
        // fp8 path via temporary fp32 view
        auto x_f32 = x_groups.to(torch::kFloat32).contiguous();
        const float* x_ptr = x_f32.data_ptr<float>();

        if (std::fabs(norm_f - 2.0f) < 1e-5f) {
            mse_scale_groups_kernel_typed<float, true>
                <<<blocks, threads, smem_bytes, stream>>>(
                    x_ptr, qmeta_ptr, G, group_size, P, maxq_f, norm_f);
        } else {
            mse_scale_groups_kernel_typed<float, false>
                <<<blocks, threads, smem_bytes, stream>>>(
                    x_ptr, qmeta_ptr, G, group_size, P, maxq_f, norm_f);
        }
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            torch::kHalf,
            torch::kBFloat16,
            dtype,
            "mse_scale_groups_packed_cuda",
            [&]() {
                using scalar_t_ = scalar_t;
                const scalar_t_* x_ptr = x_groups.data_ptr<scalar_t_>();

                if (std::fabs(norm_f - 2.0f) < 1e-5f) {
                    mse_scale_groups_kernel_typed<scalar_t_, true>
                        <<<blocks, threads, smem_bytes, stream>>>(
                            x_ptr, qmeta_ptr, G, group_size, P, maxq_f, norm_f);
                } else {
                    mse_scale_groups_kernel_typed<scalar_t_, false>
                        <<<blocks, threads, smem_bytes, stream>>>(
                            x_ptr, qmeta_ptr, G, group_size, P, maxq_f, norm_f);
                }
            }
        );
    }

    return qmeta_bytes;
}

} // namespace quant_grid

