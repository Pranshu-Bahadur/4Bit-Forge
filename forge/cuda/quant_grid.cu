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

// Constant memory for search grids (up to 1024 candidates)
__constant__ float c_p[1024];

// ---- PackBoost-Style Intrinsics ------------------------------------

__device__ __forceinline__ float fast_log2(float x) { return log2f(x); }
__device__ __forceinline__ float fast_exp2(float x) { return exp2f(x); }
__device__ __forceinline__ float fast_round(float x) { return __float2int_rn(x); }

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

// ---- Butterfly Reductions (Warp-Level) -----------------------------

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

    // 16-byte Vectorized Loads
    constexpr int bytes_per_load = 16;
    constexpr int elems_per_load = bytes_per_load / sizeof(scalar_t);
    const int4* x_vec = reinterpret_cast<const int4*>(x + base);
    int total_vecs = static_cast<int>(group_size / elems_per_load);

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

    int tail_start = total_vecs * elems_per_load;
    for (int idx = tail_start + tid; idx < group_size; idx += blockDim.x) {
        float v = static_cast<float>(x[base + idx]);
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
        m.log2_scale_fp = encode_scale_q88(s);
        float q0_clamped = fminf(fmaxf(q0, 0.0f), maxq);
        m.qzero          = static_cast<uint8_t>(lrintf(q0_clamped));
        m.flags          = symmetric ? 1 : 0;
        qmeta[g] = m;
    }
}

// ====================================================================
// 2) UNIFIED MSE SEARCH (Supports Brute Force AND 2-Stage)
// ====================================================================

template <typename scalar_t, bool IS_L2_NORM>
__global__ void mse_search_kernel(
    const scalar_t* __restrict__ x,
    QMetaPacked* __restrict__ qmeta,
    int64_t G,
    int64_t group_size,
    int64_t P_total,
    int64_t P_coarse, 
    float maxq,
    float norm
) {
    const int g = blockIdx.x;
    if (g >= G) return;

    const int lane = threadIdx.x;
    int64_t base = static_cast<int64_t>(g) * group_size;

    QMetaPacked m = qmeta[g];
    float base_s  = decode_scale_q88(m.log2_scale_fp);
    float q0      = float(m.qzero);

    extern __shared__ float cached_x[];

    // 1. Cooperative Load
    for (int64_t i = lane; i < group_size; i += 32) {
        cached_x[i] = static_cast<float>(x[base + i]);
    }
    __syncthreads(); 

    float best_loss = FLT_MAX;
    float best_s    = base_s;

    // --- STAGE 1 (Always Runs): Coarse Grid Search ---
    // If P_coarse == P_total, this is the ONLY stage (Brute Force Mode)
    
    int64_t k = 0;
    int64_t P_vec = P_coarse & ~3;

    // Unrolled by 4
    for (; k < P_vec; k += 4) {
        float p0 = c_p[k];   float p1 = c_p[k+1];
        float p2 = c_p[k+2]; float p3 = c_p[k+3];

        float s0 = base_s * p0; float rcp0 = 1.0f / s0;
        float s1 = base_s * p1; float rcp1 = 1.0f / s1;
        float s2 = base_s * p2; float rcp2 = 1.0f / s2;
        float s3 = base_s * p3; float rcp3 = 1.0f / s3;

        float l0 = 0.0f; float l1 = 0.0f;
        float l2 = 0.0f; float l3 = 0.0f;

        #pragma unroll 4
        for (int64_t i = lane; i < group_size; i += 32) {
            float v = cached_x[i];
            { float q = fminf(fmaxf(fast_round(v*rcp0 + q0), 0.0f), maxq);
              float d = fabsf((q-q0)*s0 - v); l0 += (IS_L2_NORM ? d*d : powf(d, norm)); }
            { float q = fminf(fmaxf(fast_round(v*rcp1 + q0), 0.0f), maxq);
              float d = fabsf((q-q0)*s1 - v); l1 += (IS_L2_NORM ? d*d : powf(d, norm)); }
            { float q = fminf(fmaxf(fast_round(v*rcp2 + q0), 0.0f), maxq);
              float d = fabsf((q-q0)*s2 - v); l2 += (IS_L2_NORM ? d*d : powf(d, norm)); }
            { float q = fminf(fmaxf(fast_round(v*rcp3 + q0), 0.0f), maxq);
              float d = fabsf((q-q0)*s3 - v); l3 += (IS_L2_NORM ? d*d : powf(d, norm)); }
        }

        l0 = butterflyReduceSum(l0); l1 = butterflyReduceSum(l1);
        l2 = butterflyReduceSum(l2); l3 = butterflyReduceSum(l3);

        if (lane == 0) {
            if (l0 < best_loss) { best_loss = l0; best_s = s0; }
            if (l1 < best_loss) { best_loss = l1; best_s = s1; }
            if (l2 < best_loss) { best_loss = l2; best_s = s2; }
            if (l3 < best_loss) { best_loss = l3; best_s = s3; }
        }
    }
    // Tail of Stage 1
    for (; k < P_coarse; ++k) {
        float s = base_s * c_p[k];
        float rcp = 1.0f / s;
        float loss = 0.0f;
        #pragma unroll 4
        for (int64_t i = lane; i < group_size; i += 32) {
            float v = cached_x[i];
            float q = fminf(fmaxf(fast_round(v*rcp + q0), 0.0f), maxq);
            float d = fabsf((q-q0)*s - v);
            loss += (IS_L2_NORM ? d*d : powf(d, norm));
        }
        loss = butterflyReduceSum(loss);
        if (lane == 0 && loss < best_loss) { best_loss = loss; best_s = s; }
    }

    // --- STAGE 2: FINE SEARCH (Optional) ---
    // If P_fine_count <= 0, this block is compiled away (Zero Overhead).
    
    int64_t P_fine_count = P_total - P_coarse;
    
    if (P_fine_count > 0) {
        // Sync: Broadcast best_s from Stage 1 to use as base for Stage 2
        float coarse_s = __shfl_sync(0xffffffff, best_s, 0);
        int64_t j = 0;
        
        for (; j < (P_fine_count & ~3); j += 4) {
            int64_t idx = P_coarse + j;
            float p0 = c_p[idx];   float p1 = c_p[idx+1];
            float p2 = c_p[idx+2]; float p3 = c_p[idx+3];

            // Apply fine factor relative to Coarse Winner
            float s0 = coarse_s * p0; float rcp0 = 1.0f / s0;
            float s1 = coarse_s * p1; float rcp1 = 1.0f / s1;
            float s2 = coarse_s * p2; float rcp2 = 1.0f / s2;
            float s3 = coarse_s * p3; float rcp3 = 1.0f / s3;

            float l0 = 0.0f; float l1 = 0.0f;
            float l2 = 0.0f; float l3 = 0.0f;

            #pragma unroll 4
            for (int64_t i = lane; i < group_size; i += 32) {
                float v = cached_x[i];
                { float q = fminf(fmaxf(fast_round(v*rcp0 + q0), 0.0f), maxq);
                  float d = fabsf((q-q0)*s0 - v); l0 += (IS_L2_NORM ? d*d : powf(d, norm)); }
                { float q = fminf(fmaxf(fast_round(v*rcp1 + q0), 0.0f), maxq);
                  float d = fabsf((q-q0)*s1 - v); l1 += (IS_L2_NORM ? d*d : powf(d, norm)); }
                { float q = fminf(fmaxf(fast_round(v*rcp2 + q0), 0.0f), maxq);
                  float d = fabsf((q-q0)*s2 - v); l2 += (IS_L2_NORM ? d*d : powf(d, norm)); }
                { float q = fminf(fmaxf(fast_round(v*rcp3 + q0), 0.0f), maxq);
                  float d = fabsf((q-q0)*s3 - v); l3 += (IS_L2_NORM ? d*d : powf(d, norm)); }
            }

            l0 = butterflyReduceSum(l0); l1 = butterflyReduceSum(l1);
            l2 = butterflyReduceSum(l2); l3 = butterflyReduceSum(l3);

            if (lane == 0) {
                if (l0 < best_loss) { best_loss = l0; best_s = s0; }
                if (l1 < best_loss) { best_loss = l1; best_s = s1; }
                if (l2 < best_loss) { best_loss = l2; best_s = s2; }
                if (l3 < best_loss) { best_loss = l3; best_s = s3; }
            }
        }
        for (; j < P_fine_count; ++j) {
            float s = coarse_s * c_p[P_coarse + j];
            float rcp = 1.0f / s;
            float loss = 0.0f;
            #pragma unroll 4
            for (int64_t i = lane; i < group_size; i += 32) {
                float v = cached_x[i];
                float q = fminf(fmaxf(fast_round(v*rcp + q0), 0.0f), maxq);
                float d = fabsf((q-q0)*s - v);
                loss += (IS_L2_NORM ? d*d : powf(d, norm));
            }
            loss = butterflyReduceSum(loss);
            if (lane == 0 && loss < best_loss) { best_loss = loss; best_s = s; }
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

    int threads = std::min<int64_t>(256, group_size);
    threads = align_to_warp(threads);
    if (threads < 32) threads = 32;
    const int blocks = static_cast<int>(G);

    // Calculate SMEM for build_meta (used for block reduction if threads > 32)
    const int warps_per_block = threads / 32;
    const size_t smem_bytes = 2 * static_cast<size_t>(warps_per_block) * sizeof(float);

    auto stream = at::cuda::getCurrentCUDAStream();

    // Explicit dispatch for FP8 to fix PyTest "NotImplemented" error
    if (dtype == c10::ScalarType::Float8_e4m3fn || dtype == c10::ScalarType::Float8_e4m3fnuz) {
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

    // If p_coarse_count is -1, assume user provided a mixed grid (50/50 split).
    // If user provided a pure grid, they should pass p_coarse_count = P.
    int64_t p_coarse = P / 2;

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

    const int blocks = static_cast<int>(G);
    const int threads = 32;  // Single Warp
    const size_t smem_bytes = static_cast<size_t>(group_size) * sizeof(float);

    float maxq_f = static_cast<float>(maxq);
    float norm_f = static_cast<float>(norm);

    auto dtype = x_groups.scalar_type();

    // Explicit dispatch for FP8
    if (dtype == c10::ScalarType::Float8_e4m3fn || dtype == c10::ScalarType::Float8_e4m3fnuz) {
        auto x_f32 = x_groups.to(torch::kFloat32).contiguous();
        const float* x_ptr = x_f32.data_ptr<float>();
        
        bool is_l2 = (std::fabs(norm_f - 2.0f) < 1e-5f);
        if (is_l2) {
            mse_search_kernel<float, true>
                <<<blocks, threads, smem_bytes, stream>>>(
                    x_ptr, qmeta_ptr, G, group_size, P, p_coarse, maxq_f, norm_f
                );
        } else {
            mse_search_kernel<float, false>
                <<<blocks, threads, smem_bytes, stream>>>(
                    x_ptr, qmeta_ptr, G, group_size, P, p_coarse, maxq_f, norm_f
                );
        }
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            torch::kHalf, torch::kBFloat16, dtype, "mse_scale_groups_packed_cuda",
            [&]() {
                using scalar_t_ = scalar_t;
                const scalar_t_* x_ptr = x_groups.data_ptr<scalar_t_>();
                bool is_l2 = (std::fabs(norm_f - 2.0f) < 1e-5f);

                if (is_l2) {
                    mse_search_kernel<scalar_t_, true>
                        <<<blocks, threads, smem_bytes, stream>>>(
                            x_ptr, qmeta_ptr, G, group_size, P, p_coarse, maxq_f, norm_f
                        );
                } else {
                    mse_search_kernel<scalar_t_, false>
                        <<<blocks, threads, smem_bytes, stream>>>(
                            x_ptr, qmeta_ptr, G, group_size, P, p_coarse, maxq_f, norm_f
                        );
                }
            }
        );
    }

    CUDA_CHECK(cudaGetLastError());
    return qmeta_bytes;
}