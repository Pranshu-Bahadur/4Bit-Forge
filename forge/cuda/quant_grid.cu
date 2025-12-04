#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <cmath>

// -------------------------------------------------------------
// Utilities
// -------------------------------------------------------------

#define CUDA_CHECK(err)                                                   \
  TORCH_CHECK((err) == cudaSuccess, "CUDA error: ",                       \
              cudaGetErrorString(err), " at ", __FILE__, ":", __LINE__)

static inline int ceil_div_int(int a, int b) {
  return (a + b - 1) / b;
}

static inline int align_to_warp(int threads) {
  return (threads + 31) & ~31;
}

// group_size is multiple of 32, threads_per_block is multiple of 32
static inline size_t calc_smem_gptq(int group_size, int /*threads_per_block*/) {
  // dynamic shared: cached_x[group_size]
  size_t bytes_x = static_cast<size_t>(group_size) * sizeof(float);
  // partials are in static shared (32 floats), negligible and handled by compiler
  size_t bytes_partials = 32 * sizeof(float);
  return bytes_x + bytes_partials;
}

// Choose the largest thread-block size that fits SMEM
static inline int choose_threads_that_fit(int group_size, size_t smem_cap) {
  int target = std::min(group_size, 1024);
  int threads = align_to_warp(target);
  while (threads >= 32) {
    if (calc_smem_gptq(group_size, threads) <= smem_cap) {
      return threads;
    }
    threads -= 32;
  }
  return 32;
}

// -------------------------------------------------------------
// Packed quant meta: 4 bytes per group
// -------------------------------------------------------------

struct QMeta4 {
  int16_t log2_scale_fp;  // log2(scale) * 256
  uint8_t qzero;          // zero-point (0..255)
  uint8_t flags;          // reserved / future use
};

static_assert(sizeof(QMeta4) == 4, "QMeta4 must be 4 bytes");

// log2 scale encoding in Q8.8
__device__ __forceinline__ int16_t encode_scale_q88(float scale) {
  float s = fmaxf(scale, 1e-16f);
  float l2 = __log2f(s);
  float v  = l2 * 256.0f;
  float r  = nearbyintf(v);
  if (r > 32767.0f) r = 32767.0f;
  if (r < -32768.0f) r = -32768.0f;
  return static_cast<int16_t>(r);
}

__device__ __forceinline__ float decode_scale_q88(int16_t log2_scale_fp) {
  return __exp2f(float(log2_scale_fp) * (1.0f / 256.0f));
}

// -------------------------------------------------------------
// Warp reduction helper
// -------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// -------------------------------------------------------------
// Constant memory for shrink factors p (MSE search)
// -------------------------------------------------------------

__constant__ float c_p[1024];  // up to 1024 grid points

// -------------------------------------------------------------
// Kernel 1: range-based group meta -> QMeta4
// -------------------------------------------------------------

__global__ void build_group_meta_packed_kernel(
    const float* __restrict__ x_groups,  // [G, group_size]
    QMeta4* __restrict__ qmeta,          // [G]
    int64_t G,
    int64_t group_size,
    float maxq,
    bool symmetric)
{
  int g = blockIdx.x;
  if (g >= G) return;

  int tid = threadIdx.x;
  int64_t base = static_cast<int64_t>(g) * group_size;

  extern __shared__ float smem[];  // size >= 2 * blockDim.x
  float* s_min = smem;
  float* s_max = smem + blockDim.x;

  float local_min = std::numeric_limits<float>::infinity();
  float local_max = -std::numeric_limits<float>::infinity();

  for (int i = tid; i < group_size; i += blockDim.x) {
    float v = x_groups[base + i];
    local_min = fminf(local_min, v);
    local_max = fmaxf(local_max, v);
  }

  s_min[tid] = local_min;
  s_max[tid] = local_max;
  __syncthreads();

  // block reduce
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_min[tid] = fminf(s_min[tid], s_min[tid + offset]);
      s_max[tid] = fmaxf(s_max[tid], s_max[tid + offset]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    float xmin = s_min[0];
    float xmax = s_max[0];
    float scale;
    float qzero;

    if (symmetric) {
      float amax = fmaxf(fabsf(xmin), fabsf(xmax));
      scale = (2.0f / maxq) * amax + 1e-12f;
      qzero = (maxq + 1.0f) * 0.5f;
    } else {
      scale = (xmax - xmin) / maxq + 1e-12f;
      qzero = -xmin / scale;
      if (qzero < 0.0f) qzero = 0.0f;
      if (qzero > maxq) qzero = maxq;
    }

    QMeta4 m;
    m.log2_scale_fp = encode_scale_q88(scale);
    m.qzero         = static_cast<uint8_t>(fminf(fmaxf(qzero, 0.0f), maxq));
    m.flags         = 0;
    qmeta[g]        = m;
  }
}

// -------------------------------------------------------------
// Kernel 2: MSE scale refinement (grid search over c_p)
// -------------------------------------------------------------

template <bool IS_L2>
__global__ void mse_scale_groups_packboost_kernel(
    const float* __restrict__ x_groups,  // [G, group_size]
    QMeta4* __restrict__ qmeta,          // [G]
    int64_t G,
    int64_t group_size,
    int64_t P,
    float maxq,
    float norm)
{
  int g = blockIdx.x;
  if (g >= G) return;

  int tid = threadIdx.x;
  int64_t base = static_cast<int64_t>(g) * group_size;

  extern __shared__ float cached_x[];  // size >= group_size
  __shared__ float shared_partials[32];  // max 32 warps (1024 threads)

  // 1. Load group into shared once
  for (int i = tid; i < group_size; i += blockDim.x) {
    cached_x[i] = x_groups[base + i];
  }
  __syncthreads();

  // 2. Load base meta into registers
  QMeta4 m = qmeta[g];
  float base_s = decode_scale_q88(m.log2_scale_fp);
  float q0     = float(m.qzero);

  float best_loss = std::numeric_limits<float>::infinity();
  float best_s    = base_s;

  int nwarps = blockDim.x / 32;
  int lane   = tid & 31;
  int wid    = tid >> 5;

  // 3. Grid search over shrink factors
  for (int k = 0; k < P; ++k) {
    float shrink = c_p[k];
    float s = base_s * shrink;
    if (s <= 0.0f) continue;

    float local_loss = 0.0f;

    for (int i = tid; i < group_size; i += blockDim.x) {
      float v = cached_x[i];

      float q = nearbyintf(v / s + q0);
      if (q < 0.0f)   q = 0.0f;
      if (q > maxq)   q = maxq;

      float y    = (q - q0) * s;
      float diff = fabsf(y - v);

      if (IS_L2) {
        local_loss += diff * diff;
      } else {
        local_loss += powf(diff, norm);
      }
    }

    // warp-level reduce
    local_loss = warpReduceSum(local_loss);

    if (lane == 0) {
      shared_partials[wid] = local_loss;
    }
    __syncthreads();

    // first warp reduces partials
    if (wid == 0) {
      float val = (lane < nwarps) ? shared_partials[lane] : 0.0f;
      val = warpReduceSum(val);
      if (lane == 0) {
        if (val < best_loss) {
          best_loss = val;
          best_s    = s;
        }
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    m.log2_scale_fp = encode_scale_q88(best_s);
    qmeta[g]        = m;
  }
}

// -------------------------------------------------------------
// Host wrappers (called from kernels.cpp)
// -------------------------------------------------------------

// Build initial group meta (range-based) as packed bytes + maxq scalar
std::tuple<torch::Tensor, torch::Tensor> build_group_meta_packed_cuda(
    torch::Tensor x_groups,
    int64_t bit_width,
    bool symmetric)
{
  TORCH_CHECK(x_groups.is_cuda(), "x_groups must be CUDA");
  TORCH_CHECK(x_groups.dim() == 2, "x_groups must be [G, group_size]");
  TORCH_CHECK(x_groups.dtype() == torch::kFloat32,
              "x_groups must be float32 (cast in Python before calling)");

  auto G          = x_groups.size(0);
  auto group_size = x_groups.size(1);
  TORCH_CHECK(group_size % 32 == 0,
              "group_size must be a multiple of 32");

  int maxq_val = (1 << bit_width) - 1;
  auto device  = x_groups.device();

  auto stream = c10::cuda::getCurrentCUDAStream();

  torch::Tensor qmeta_bytes =
      torch::empty({G, 4}, torch::dtype(torch::kUInt8).device(device));

  torch::Tensor maxq =
      torch::tensor(float(maxq_val),
                    torch::dtype(torch::kFloat32).device(device));

  const float* x_ptr = x_groups.data_ptr<float>();
  auto* qmeta_ptr =
      reinterpret_cast<QMeta4*>(qmeta_bytes.data_ptr<uint8_t>());

  auto* prop = at::cuda::getCurrentDeviceProperties();
  size_t smem_cap = prop->sharedMemPerBlockOptin
                        ? size_t(prop->sharedMemPerBlockOptin)
                        : size_t(prop->sharedMemPerBlock);

  int threads = choose_threads_that_fit((int)group_size, smem_cap);
  size_t smem_bytes =
      2 * size_t(threads) * sizeof(float);  // s_min + s_max arrays

  dim3 blocks((unsigned int)G);
  dim3 tpb((unsigned int)threads);

  build_group_meta_packed_kernel<<<blocks, tpb, smem_bytes, stream>>>(
      x_ptr,
      qmeta_ptr,
      G,
      group_size,
      maxq.item<float>(),
      symmetric);

  return {qmeta_bytes, maxq};
}

// MSE search to refine scales stored in QMeta4
torch::Tensor mse_scale_groups_packed_cuda(
    torch::Tensor x_groups,
    torch::Tensor p,
    torch::Tensor qmeta_bytes,
    double maxq,
    double norm)
{
  TORCH_CHECK(x_groups.is_cuda(), "x_groups must be CUDA");
  TORCH_CHECK(p.is_cuda(),        "p must be CUDA");
  TORCH_CHECK(qmeta_bytes.is_cuda(), "qmeta_bytes must be CUDA");

  TORCH_CHECK(x_groups.dim() == 2, "x_groups must be [G, group_size]");
  TORCH_CHECK(x_groups.dtype() == torch::kFloat32,
              "x_groups must be float32 (cast in Python before calling)");
  TORCH_CHECK(qmeta_bytes.dim() == 2 &&
              qmeta_bytes.size(1) == 4 &&
              qmeta_bytes.dtype() == torch::kUInt8,
              "qmeta_bytes must be [G,4] uint8");

  auto G          = x_groups.size(0);
  auto group_size = x_groups.size(1);
  auto P          = p.size(0);

  TORCH_CHECK(group_size % 32 == 0,
              "group_size must be a multiple of 32");
  TORCH_CHECK(P <= 1024,
              "P (quant_n_grid) must be <= 1024 for c_p constant cache");

  auto device = x_groups.device();
  auto stream = c10::cuda::getCurrentCUDAStream();

  // copy shrink factors into constant memory
  CUDA_CHECK(cudaMemcpyToSymbol(
      c_p,
      p.data_ptr<float>(),
      static_cast<size_t>(P) * sizeof(float),
      0,
      cudaMemcpyDeviceToDevice));

  auto* prop = at::cuda::getCurrentDeviceProperties();
  size_t smem_cap = prop->sharedMemPerBlockOptin
                        ? size_t(prop->sharedMemPerBlockOptin)
                        : size_t(prop->sharedMemPerBlock);

  int threads = choose_threads_that_fit((int)group_size, smem_cap);
  size_t smem_bytes = size_t(group_size + 32) * sizeof(float);  // cached_x + headroom

  dim3 blocks((unsigned int)G);
  dim3 tpb((unsigned int)threads);

  const float* x_ptr = x_groups.data_ptr<float>();
  auto* qmeta_ptr =
      reinterpret_cast<QMeta4*>(qmeta_bytes.data_ptr<uint8_t>());

  float maxq_f = static_cast<float>(maxq);
  float norm_f = static_cast<float>(norm);

  if (std::abs(norm_f - 2.0f) < 1e-5f) {
    mse_scale_groups_packboost_kernel<true>
        <<<blocks, tpb, smem_bytes, stream>>>(
            x_ptr,
            qmeta_ptr,
            G,
            group_size,
            P,
            maxq_f,
            norm_f);
  } else {
    mse_scale_groups_packboost_kernel<false>
        <<<blocks, tpb, smem_bytes, stream>>>(
            x_ptr,
            qmeta_ptr,
            G,
            group_size,
            P,
            maxq_f,
            norm_f);
  }

  return qmeta_bytes;
}
