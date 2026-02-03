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


__global__ void sparse18_unstructured_gemm_kernel(
    uint32_t* __restrict__ qWpacked_u32, //G, R | G = (ceil(C/32)) | idx4 + int4
    __nv_bfloat16*  X, //N, C
    __nv_bfloat16* __restrict__ Y, //N, R
    const float*  scales, //G, R | G = (ceil(C/32))
    const uint8_t* qzeros, //G, R | G = (ceil(C/32))
    int64_t N,
    int64_t R,
    int64_t C,
    int64_t G
) {
    int64_t lane = threadIdx.x & 31;
    int64_t wid = ((blockIdx.x * blockDim.x) + threadIdx.x) >> 5; // global warp id
    int64_t RT = (R + 31) >> 5;
    int64_t rt = wid % RT;
    int64_t ntile = wid / RT;      // warp-uniform
    int64_t n    = ntile * 4;     // base row for this warp
    if (n >= N) return;
    int64_t r = (rt << 5) + lane;
    float x0, x1, x2, x3;
    float v0, v1, v2, v3;
    
    v0 = 0.0f;
    v1 = 0.0f;
    v2 = 0.0f;
    v3 = 0.0f;

    float deqW = 0.0f;
    uint32_t qWidx4int4 = 0u;
    uint16_t idx4 = 0u;
    uint16_t qW4 = 0u;
    unsigned mask = 0xffffffff;
    float s = 0.0f;
    uint8_t q0 = 0;
    float q0s = 0.0f;
    bool active = (r < R);

    for (int64_t gid = 0; gid < G; ++gid) {
        idx4 = 0u;
        s = 0.0f;
        q0 = 0;
        qW4 = 0;
        q0s = 0.0f;
        if (active) {
                qWidx4int4 = qWpacked_u32[gid * R + r];
                idx4 = (uint16_t)(qWidx4int4 >> 16);
                qW4 = (uint16_t)(qWidx4int4 & 0xFFFF);
                s = scales[gid * R + r];
                q0 = qzeros[gid * R + r];
                q0s = ((float)q0)*s;
        }

        int64_t cid = gid << 5;

        x0 = 0.0f;
        x1 = 0.0f;
        x2 = 0.0f;
        x3 = 0.0f;

        if (cid + lane < C && (n < N)) {
                x0 = __bfloat162float(X[(n) * C + cid + lane]);
        }
        if (cid + lane < C && (n + 1 < N)) {
                x1 = __bfloat162float(X[(n + 1) * C + cid + lane]);
        }
        if (cid + lane < C && (n + 2 < N)) {
                x2 = __bfloat162float(X[(n + 2) * C + cid + lane]);
        }
        if (cid + lane < C && (n + 3 < N)) {
                x3 = __bfloat162float(X[(n + 3) * C + cid + lane]);
        }
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            deqW = (float)((int)(((qW4) >> (4 * i)) & 0xF))*s - q0s;
            uint32_t src = (i << 3) + ((idx4 >> (4 * i)) & 0xF);
            v0 += deqW * __shfl_sync(mask, x0, src);
            v1 += deqW * __shfl_sync(mask, x1, src);
            v2 += deqW * __shfl_sync(mask, x2, src);
            v3 += deqW * __shfl_sync(mask, x3, src);
        }
    }
    if (active) {
        if (n < N) {
                Y[(n) * R + r] = __float2bfloat16(v0);
        }
        if (n + 1 < N) {
                Y[(n + 1) * R + r] = __float2bfloat16(v1);
        }
        if (n + 2 < N) {
                Y[(n + 2) * R + r] = __float2bfloat16(v2);
        }
        if (n + 3 < N) {
                Y[(n + 3) * R + r] = __float2bfloat16(v3);
        }
    }
}


/*
int64_t warps_per_row = (R + 31) / 32;
int64_t rows_of_warps = (N + 3) / 4; // Chunk size 4
dim3 gridDim(rows_of_warps * warps_per_row, 1, 1);
*/