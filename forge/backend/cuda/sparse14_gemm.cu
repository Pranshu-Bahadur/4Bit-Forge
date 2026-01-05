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

////////////////////////////////////////////////////////////////////////////////
// bf16 bitcast helpers
////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ uint16_t f32_to_bf16_bits(float x) {
    __nv_bfloat16 b = __float2bfloat16_rn(x);
    union { __nv_bfloat16 b; uint16_t u; } cvt;
    cvt.b = b;
    return cvt.u;
}

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t bits) {
    union { __nv_bfloat16 b; uint16_t u; } cvt;
    cvt.u = bits;
    return __bfloat162float(cvt.b);
}

__device__ __forceinline__ void unpack_sparsegptq14_u64(
    uint64_t packed,
    uint32_t& qw32,
    uint16_t& idx16,
    float& s
) {
    qw32  = (uint32_t)(packed & 0xFFFFFFFFull);
    idx16 = (uint16_t)((packed >> 32) & 0xFFFFull);
    uint16_t s_bf16 = (uint16_t)((packed >> 48) & 0xFFFFull);
    s = bf16_bits_to_f32(s_bf16);
}

// Example dequant inside matmul (symmetric, uint4 with q0 = 8 for 4-bit):
// w = (q - 8) * s
__device__ __forceinline__ float dequant_sym_u4(uint32_t q4, float s) {
    return (float((int)q4) - 8.0f) * s;
}


////////////////////////////////////////////////////////////////////////////////
// Pack format (1:4 inside 32)
// packed_u64 layout:
//
// [63:48] scale_bf16_bits | 1 scale every 64 channels
// [47:32] idx16  (8 groups * 2 bits) | each "group" has 1 value
// [31: 0] qw32   (8 groups * 4 bits) | each "group" has 1 value
//
////////////////////////////////////////////////////////////////////////////////

// Unstructured 1 out of 4 Sparsity MatMul
__global__ void sparse14_gemm(
    const uint4* __restrict__ qW2S1u64vec2, //ulong2 [G2, R]
    const float4* __restrict__ X,       // [N4, C] | N4 = ceil(N/4)
    float4* __restrict__ Y,             // [N4, R] | N4 = ceil(N/4) __nv_bfloat16
    const int64_t N,
    const int64_t R,
    const int64_t C,
    const int64_t G2, // G = ceil(C/32) | G2 = ceil(C/64)
    const int64_t NT
) {
    int64_t ntile = (blockIdx.x * blockDim.x);
    int64_t nid   = ntile * NT;
    if (nid >= N) return;

    int64_t tid = threadIdx.x;
    int64_t rid = (blockIdx.y * blockDim.x) + tid;
    int64_t lane = tid & 31;

    
}


torch::Tensor unstructured_sparse14_int4symq__gemm(
    torch::Tensor qW2S1u64, // [G, R] | G=ceil(C/32) | uint64 | Packing format defined above
    torch::Tensor X // [N, C] | bfloat16
) {

    qW2S1u64 = qW2S1u64.contiguous();
    const int64_t G = qW2S1u64.size(0);
    const int64_t R = qW2S1u64.size(1); // out_features


    X = X.contiguous();
    const int64_t N = X.size(0); // Batch Size = B*Tokens??
    const int64_t C = X.size(1); // in_features / channels

    auto Y = torch::empty({N, R}, torch::TensorOptions().dtype(torch::kBFloat16).device(X.device()));
    auto stream = at::cuda::getCurrentCUDAStream();

    const int NT = 4; // Y[n:nrows, :] or Ntile_length

    dim3 block(128);
    dim3 grid(
        (static_cast<int>(N) + (NT - 1)) / NT,
        (static_cast<int>(R) + (block.x - 1)) / block.x
    );


}
