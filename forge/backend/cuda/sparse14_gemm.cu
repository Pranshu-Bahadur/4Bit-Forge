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
