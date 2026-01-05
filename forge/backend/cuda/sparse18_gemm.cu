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


/*

#include <torch/extension.h>
#include <vector>

// Forward declaration of your CUDA kernel
void launch_sparse18_kernel(
    const uint32_t* qW, const at::Half* X, at::Half* Y, 
    const float* scales, const uint8_t* qzeros,
    int64_t N, int64_t R, int64_t C, int64_t G
);

void sparse_gemm_forward(
    torch::Tensor qW, torch::Tensor X, torch::Tensor Y, 
    torch::Tensor scales, torch::Tensor qzeros
) {
    const int64_t N = X.size(0);
    const int64_t C = X.size(1);
    const int64_t R = Y.size(1);
    const int64_t G = qW.size(0);

    launch_sparse18_kernel(
        (uint32_t*)qW.data_ptr(),
        (at::Half*)X.data_ptr(),
        (at::Half*)Y.data_ptr(),
        (float*)scales.data_ptr(),
        (uint8_t*)qzeros.data_ptr(),
        N, R, C, G
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sparse_gemm_forward, "Sparse 1:8 GEMM");
}
*/




#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Helper to process one 32-bit integer (one 32-channel group)
// We inline this to keep the main kernel readable.
__device__ __forceinline__ void process_subgroup(
    uint32_t packed_int, 
    float s, float q0s,
    const __nv_bfloat16* __restrict__ X,
    int64_t base_n, int64_t C, int64_t cid, int lane, int64_t N,
    float& v0, float& v1, float& v2, float& v3
) {
    uint16_t idx4 = (uint16_t)(packed_int >> 16);
    uint16_t qW4  = (uint16_t)(packed_int & 0xFFFF);
    unsigned mask = 0xffffffff;

    // Load X (Fetcher) - ILP=4
    float x0=0.f, x1=0.f, x2=0.f, x3=0.f;
    int64_t abs_col = cid + lane;
    
    // Bounds check column
    if (abs_col < C) {
        if (base_n + 0 < N) x0 = __bfloat162float(X[(base_n + 0) * C + abs_col]);
        if (base_n + 1 < N) x1 = __bfloat162float(X[(base_n + 1) * C + abs_col]);
        if (base_n + 2 < N) x2 = __bfloat162float(X[(base_n + 2) * C + abs_col]);
        if (base_n + 3 < N) x3 = __bfloat162float(X[(base_n + 3) * C + abs_col]);
    }

    // Compute (Solver)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float deqW = (float)((int)((qW4 >> (4 * i)) & 0xF)) * s - q0s;
        uint32_t src = (i << 3) + ((idx4 >> (4 * i)) & 0xF);
        
        v0 += deqW * __shfl_sync(mask, x0, src);
        v1 += deqW * __shfl_sync(mask, x1, src);
        v2 += deqW * __shfl_sync(mask, x2, src);
        v3 += deqW * __shfl_sync(mask, x3, src);
    }
}

__global__ void sparse18_diamond_kernel(
    const uint4* __restrict__ qW_swizzled_128, // [G_vecs, R] -> The Semi-Truck Payload
    const __nv_bfloat16* __restrict__ X,       // [N, C]
    __nv_bfloat16* __restrict__ Y,             // [N, R]
    const float* __restrict__ scales,          // [G_vecs, R] (Assumes GroupSize=128)
    const uint8_t* __restrict__ qzeros,        // [G_vecs, R]
    int64_t N, 
    int64_t R, 
    int64_t C, 
    int64_t G_vecs // = G / 4
) {
    // 1. Thread Setup
    int lane = threadIdx.x & 31;
    int64_t wid = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    // Process N in chunks of 4 rows per warp
    int64_t RT = (R + 31) >> 5;
    int64_t rt = wid % RT;
    int64_t ntile = wid / RT;
    int64_t n = ntile * 4;

    if (n >= N) return;

    int64_t r = (rt << 5) + lane;
    bool active = (r < R);

    // 2. Accumulators
    float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;

    // 3. Main Loop: Iterate over VECTORS (1 vector = 128 input channels)
    for (int64_t gv = 0; gv < G_vecs; ++gv) {
        
        // A. Vectorized Load (LDG.128)
        // Fetches 16 bytes (4 integers) in one instruction
        // Since input is [G_vecs, R], accessing [gv*R + r] is perfectly coalesced.
        uint4 vec_w = make_uint4(0,0,0,0);
        if (active) vec_w = qW_swizzled_128[gv * R + r]; 

        // B. Metadata Load (One Scale to rule them all)
        // Group Size = 128 means 1 scale applies to the entire vec_w
        float s = 0.0f;
        float q0s = 0.0f;
        if (active) {
            int64_t s_idx = gv * R + r;
            s = scales[s_idx];
            uint8_t q0 = qzeros[s_idx];
            q0s = ((float)q0) * s;
        }

        // C. Process the 4 Sub-Groups (Instruction Level Parallelism)
        // Each .x, .y, .z, .w corresponds to a 32-channel block
        
        int64_t base_cid = (gv * 4) << 5; // Start channel for this vector

        // Subgroup 0 (.x)
        process_subgroup(vec_w.x, s, q0s, X, n, C, base_cid + 0, lane, N, v0, v1, v2, v3);
        
        // Subgroup 1 (.y)
        process_subgroup(vec_w.y, s, q0s, X, n, C, base_cid + 32, lane, N, v0, v1, v2, v3);
        
        // Subgroup 2 (.z)
        process_subgroup(vec_w.z, s, q0s, X, n, C, base_cid + 64, lane, N, v0, v1, v2, v3);
        
        // Subgroup 3 (.w)
        process_subgroup(vec_w.w, s, q0s, X, n, C, base_cid + 96, lane, N, v0, v1, v2, v3);
    }

    // 4. Final Write
    if (active) {
        if (n + 0 < N) Y[(n + 0) * R + r] = __float2bfloat16(v0);
        if (n + 1 < N) Y[(n + 1) * R + r] = __float2bfloat16(v1);
        if (n + 2 < N) Y[(n + 2) * R + r] = __float2bfloat16(v2);
        if (n + 3 < N) Y[(n + 3) * R + r] = __float2bfloat16(v3);
    }
}
