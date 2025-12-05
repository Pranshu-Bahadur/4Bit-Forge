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
    int16_t  log2_scale_fp;
    uint8_t  qzero;
    uint8_t  flags;
};

// Constant memory for shrink factors (fastest broadcast access)
// Supports up to 1024 grid search candidates
__constant__ float c_p[1024];

// ---- small helpers -------------------------------------------------

__device__ __forceinline__ float fast_log2(float x) { 
    return log2f(x); 
}

__device__ __forceinline__ float fast_exp2(float x) { 
    return exp2f(x); 
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

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

inline torch::Tensor ensure_contiguous_same_dtype(torch::Tensor x) {
    if (!x.is_contiguous()) x = x.contiguous();
    return x;
}

static inline int align_to_warp(int threads) {
    return (threads + 31) & ~31;
}

// ====================================================================
// 1) ABSMAX / RANGE META (vectorized loader + warp/block reduction)
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

    // 16-byte vectorized loads via int4 for maximum bandwidth
    constexpr int bytes_per_load = 16;
    constexpr int elems_per_load = bytes_per_load / sizeof(scalar_t);

    const int4* x_vec = reinterpret_cast<const int4*>(x + base);
    int total_vecs = static_cast<int>(group_size / elems_per_load);

    // 1) Full 16-byte chunks
    for (int i = tid; i < total_vecs; i += blockDim.x) {
        int4 packed = x_vec[i];
        const scalar_t* vals = reinterpret_cast<const scalar_t*>(&packed);

        #pragma unroll
        for (int k = 0; k < elems_per_load; ++k) {
            float v = static_cast<float>(vals[k]);
            local_min = fminf(local_min, v);
            local_max = fmaxf(local_max, v);
        }
    }

    // 2) Tail
    int tail_start = total_vecs * elems_per_load;
    for (int idx = tail_start + tid; idx < group_size; idx += blockDim.x) {
        float v = static_cast<float>(x[base + idx]);
        local_min = fminf(local_min, v);
        local_max = fmaxf(local_max, v);
    }

    // 3) Reduce
    local_min = warpReduceMin(local_min);
    local_max = warpReduceMax(local_max);

    extern __shared__ float sdata[];
    int warps_per_block = blockDim.x / 32;
    float* smin = sdata;
    float* smax = sdata + warps_per_block;

    int lane   = tid & 31;
    int warpId = tid >> 5;

    if (lane == 0) {
        smin[warpId] = local_min;
        smax[warpId] = local_max;
    }
    __syncthreads();

    if (warpId == 0) {
        float red_min = (tid < warps_per_block) ? smin[tid] : 1e30f;
        float red_max = (tid < warps_per_block) ? smax[tid] : -1e30f;

        red_min = warpReduceMin(red_min);
        red_max = warpReduceMax(red_max);

        if (tid == 0) {
            float xmin = red_min;
            float xmax = red_max;
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
            m.flags          = symmetric ? 1u : 0u;
            qmeta[g] = m;
        }
    }
}

// ====================================================================
// 2) MULTI-WARP MSE SEARCH (Optimized for group_size=128)
// ====================================================================

template <typename scalar_t, bool IS_L2_NORM>
__global__ void mse_scale_groups_multiwarp(
    const scalar_t* __restrict__ x,
    QMetaPacked* __restrict__ qmeta,
    int64_t G,
    int64_t group_size,
    int64_t P,
    float maxq,
    float norm
) {
    // We launch 1 Block per Group.
    // Recommended threads = 128 (4 warps) for group_size=128.
    const int g = blockIdx.x;
    if (g >= G) return;

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int num_warps = blockDim.x / 32;

    int64_t base = static_cast<int64_t>(g) * group_size;

    QMetaPacked m = qmeta[g];
    float base_s  = decode_scale_q88(m.log2_scale_fp);
    float q0      = float(m.qzero);

    // Dynamic Shared Memory Layout
    // [0..num_warps-1]: best_loss per warp
    // [num_warps..2*num_warps-1]: best_scale per warp
    // [2*num_warps...]: cached group data
    extern __shared__ float smem[];
    float* smem_loss  = smem;
    float* smem_scale = smem + num_warps;
    float* cached_x   = smem_scale + num_warps;

    // 1. Cooperative Load
    // For group_size=128 and blockDim=128, this is a perfect 1:1 copy.
    for (int i = tid; i < group_size; i += blockDim.x) {
        cached_x[i] = static_cast<float>(x[base + i]);
    }
    __syncthreads();

    // 2. Parallel Grid Search
    // Warps split the candidates P amongst themselves.
    float best_loss = FLT_MAX;
    float best_s    = base_s;

    for (int64_t k = warp; k < P; k += num_warps) {
        float shrink = c_p[k];
        float s = base_s * shrink;
        if (s <= 1e-12f) continue;

        float rcp_s = 1.0f / s;
        float iter_loss = 0.0f;

        // Inner loop: each warp computes loss for ONE candidate 's'
        // across the entire group.
        // #pragma unroll 4 is standard optimal for float loads
        #pragma unroll 4
        for (int i = lane; i < group_size; i += 32) {
            float v = cached_x[i]; // Broadcast from SMEM
            
            // Quantize
            float val = v * rcp_s + q0;
            float q   = rintf(val);
            q = fminf(fmaxf(q, 0.0f), maxq);
            
            // Dequant & Error
            float diff = fabsf((q - q0) * s - v);
            
            if constexpr (IS_L2_NORM) iter_loss += diff * diff;
            else iter_loss += powf(diff, norm);
        }

        // Reduce across lanes to get total error for candidate 'k'
        iter_loss = warpReduceSum(iter_loss);

        // Update Warp-Local Best
        if (lane == 0) {
            if (iter_loss < best_loss) {
                best_loss = iter_loss;
                best_s    = s;
            }
        }
    }

    // 3. Block Reduction (Reduction of Warps)
    if (lane == 0) {
        smem_loss[warp]  = best_loss;
        smem_scale[warp] = best_s;
    }
    __syncthreads();

    // Warp 0, Lane 0 determines the Global Winner
    if (tid == 0) {
        float global_best_loss = smem_loss[0];
        float global_best_s    = smem_scale[0];

        #pragma unroll
        for (int w = 1; w < num_warps; ++w) {
            float l = smem_loss[w];
            if (l < global_best_loss) {
                global_best_loss = l;
                global_best_s    = smem_scale[w];
            }
        }

        m.log2_scale_fp = encode_scale_q88(global_best_s);
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

    // launch config
    int threads = std::min<int64_t>(256, group_size);
    threads = align_to_warp(threads);
    if (threads < 32) threads = 32;
    const int blocks = static_cast<int>(G);

    const int warps_per_block = threads / 32;
    const size_t smem_bytes = 2 * static_cast<size_t>(warps_per_block) * sizeof(float);

    auto stream = at::cuda::getCurrentCUDAStream();

    if (dtype == torch::kFloat8_e4m3fn || dtype == torch::kFloat8_e4m3fnuz) {
        auto x_f32 = x_groups.to(torch::kFloat32).contiguous();
        const float* x_ptr = x_f32.data_ptr<float>();
        build_group_meta_optimized<float>
            <<<blocks, threads, smem_bytes, stream>>>(
                x_ptr, qmeta_ptr, G, group_size, static_cast<int>(bit_width), symmetric
            );
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            torch::kHalf, torch::kBFloat16, dtype, "build_group_meta_packed_cuda",
            [&]() {
                using scalar_t_ = scalar_t;
                const scalar_t_* x_ptr = x_groups.data_ptr<scalar_t_>();
                build_group_meta_optimized<scalar_t_>
                    <<<blocks, threads, smem_bytes, stream>>>(
                        x_ptr, qmeta_ptr, G, group_size, static_cast<int>(bit_width), symmetric
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

    // MULTI-WARP CONFIGURATION
    // For group_size=128, we use 128 threads (4 warps) per block.
    // This provides 4x parallelism for the Grid Search (P)
    int threads = (group_size >= 128) ? 128 : static_cast<int>(group_size);
    threads = align_to_warp(threads);
    if (threads < 32) threads = 32;
    
    const int blocks = static_cast<int>(G);
    const int num_warps = threads / 32;

    // Shared Mem: [best_loss(warps) | best_scale(warps) | cached_x(group_size)]
    const size_t smem_bytes = (2 * num_warps + group_size) * sizeof(float);

    float maxq_f = static_cast<float>(maxq);
    float norm_f = static_cast<float>(norm);

    if (dtype == torch::kFloat8_e4m3fn || dtype == torch::kFloat8_e4m3fnuz) {
        auto x_f32 = x_groups.to(torch::kFloat32).contiguous();
        const float* x_ptr = x_f32.data_ptr<float>();

        if (fabsf(norm_f - 2.0f) < 1e-5f) {
            mse_scale_groups_multiwarp<float, true><<<blocks, threads, smem_bytes, stream>>>(
                    x_ptr, qmeta_ptr, G, group_size, P, maxq_f, norm_f
            );
        } else {
            mse_scale_groups_multiwarp<float, false><<<blocks, threads, smem_bytes, stream>>>(
                    x_ptr, qmeta_ptr, G, group_size, P, maxq_f, norm_f
            );
        }
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            torch::kHalf, torch::kBFloat16, dtype, "mse_scale_groups_packed_cuda",
            [&]() {
                using scalar_t_ = scalar_t;
                const scalar_t_* x_ptr = x_groups.data_ptr<scalar_t_>();

                if (fabsf(norm_f - 2.0f) < 1e-5f) {
                    mse_scale_groups_multiwarp<scalar_t_, true><<<blocks, threads, smem_bytes, stream>>>(
                            x_ptr, qmeta_ptr, G, group_size, P, maxq_f, norm_f
                    );
                } else {
                    mse_scale_groups_multiwarp<scalar_t_, false><<<blocks, threads, smem_bytes, stream>>>(
                            x_ptr, qmeta_ptr, G, group_size, P, maxq_f, norm_f
                    );
                }
            }
        );
    }

    CUDA_CHECK(cudaGetLastError());
    return qmeta_bytes;
}