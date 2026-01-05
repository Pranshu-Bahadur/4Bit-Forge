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
#include <stdint.h>

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

////////////////////////////////////////////////////////////////////////////////
// Pack format (1:4 inside 32)
// packed_u64 layout:
//
// [63:48] scale_bf16_bits
// [47:32] idx16  (8 groups * 2 bits)
// [31: 0] qw32   (8 groups * 4 bits)
//
////////////////////////////////////////////////////////////////////////////////

__global__ void pack_sparsegptq14_u64_kernel(
    const uint8_t* __restrict__ qweight,   // [C, R], low 4 bits valid
    const uint32_t* __restrict__ M,        // [G32, R], G32 = ceil(C/32)
    const float* __restrict__ scales,      // [G32, R]  (symmetric => no qzeros needed)
    uint64_t* __restrict__ qWpack_u64,     // [G32, R]
    int64_t C,
    int64_t R,
    int64_t G32
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t r   = tid % R;
    int64_t gid = tid / R;
    if (r >= R || gid >= G32) return;

    uint32_t bitmask = M[gid * R + r];

    // idx16: 8 * idx2 (0..3)
    uint16_t idx16 = 0;
    // qw32 : 8 * int4 (0..15)
    uint32_t qw32  = 0;

    // Each 32-chunk has 8 subgroups of 4 channels: [0..3],[4..7],...,[28..31]
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t sub = (bitmask >> (4 * i)) & 0xFu;  // 4-bit subgroup mask
        if (sub == 0u) continue;                     // tail safety

        // For true 1:4, sub should be one-hot. We take the least-significant set bit.
        int c = __ffs(sub) - 1;                      // 0..3
        idx16 |= (uint16_t)(c & 0x3) << (2 * i);

        int64_t cid = gid * 32 + i * 4 + c;
        if (cid < C) {
            uint32_t q = (uint32_t)(qweight[cid * R + r] & 0xFu);
            qw32 |= (q << (4 * i));
        }
    }

    // Pack bf16 scale bits
    float s = scales[gid * R + r];
    uint16_t s_bf16 = f32_to_bf16_bits(s);

    qWpack_u64[gid * R + r] =
        ((uint64_t)s_bf16 << 48) |
        ((uint64_t)idx16  << 32) |
        (uint64_t)qw32;
}

////////////////////////////////////////////////////////////////////////////////
// Decode snippet (for matmul)
////////////////////////////////////////////////////////////////////////////////
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

// Example per-(gid32,r) accumulate for 1 row x:
// for i=0..7:
//   src = 4*i + idx2
//   q   = (qw32 >> (4*i)) & 0xF
//   w   = (q - 8) * s
//   acc += w * shfl(x, src)
