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

__device__ __forceinline__ uint16_t packidx4(uint32_t bitmask) {
    uint16_t idx4 = 0u;
    #pragma unroll
    for (int i=0; i < 4; ++i) {
        uint32_t mask = (bitmask >> (i*8)) & 0xFF;
        uint32_t keep = __popc(mask - 1);
        if (mask != 0u && keep < 8u) {
            idx4 |= (keep << (4*i));
        }
    }
    return idx4;
}


__global__ void bitpack_sparsegptq18_kernel(
    uint8_t* __restrict__ qweight, //C, R
    uint32_t* __restrict__ M, //KT, R| KT = ceil(C/32)
    uint32_t* __restrict__ qWpack_u32, //KT, R
    int64_t C, 
    int64_t R,
    int64_t KT
) {

    int64_t tid = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x); //global thread id
    int64_t r = tid % R;
    int64_t k = tid / R;
    if (r >= R || k >= KT) return;

    uint32_t bitmask = M[k * R + r];
    uint16_t idx4 = packidx4(bitmask);
    uint16_t qw4 = 0u;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t submask = (bitmask >> (i * 8)) & 0xFF;
        if (submask==0) {
            continue;
        }
        uint8_t c = idx4 >> (4*i) & 0x0F;
        int cid = ((k * 32) + (i * 8) + c);
        if (cid < C) {
           qw4 |= (uint16_t)(qweight[cid * R + r]  & 0x0F) << (4*i);
        }
    }
    qWpack_u32[k * R + r] = (uint32_t)(idx4 << 16) | (uint32_t)qw4;
}