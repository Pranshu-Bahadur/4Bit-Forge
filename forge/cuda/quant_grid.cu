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

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
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
__device__ __forceinline__ T warpReduceMin(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void build_quantization_grid(
    const float* __restrict__ X, //R*G, group_size
    float* scales, //R*G
    float* qzeros, //R*G
    const int64_t  RG,
    const int64_t  group_size,
    const int  bit_width,
    const bool  symmetric
) {
    
    int bid = blockIdx.x;
    int wpb = blockDim.x / 32; // 4
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wib = tid / 32; //warps in block
    int wid = (int64_t)bid * wpb + wib;


    for (int64_t g = wid; g < (int64_t)(RG); g += (int64_t)(gridDim.x * wpb)) {
        const float* group = X + (g*group_size);
        float val = 0.0f;
        float local_min = 1e30f;
        float local_max = -1e30f;

        for (int i = lane; i < group_size; i += 32) {
            val = group[i];
            local_max = fmaxf(local_max, val);
            local_min = fminf(local_min, val);
        }
        local_max = warpReduceMax(local_max);
        local_min = warpReduceMin(local_min);
        
        if (lane == 0) {
            float xmin = local_min;
            float xmax = local_max;
            float maxq = float((1 << bit_width) - 1);
            float eps  = 1e-12f;
            float s, q0;

            if (symmetric) {
                float amax = fmaxf(fabsf(xmin), fabsf(xmax));
                s  = (2.0f / (maxq)) * amax + eps;
                q0 = 0.5f * (maxq + 1.0f);
            } else {
                s  = (xmax - xmin) / maxq + eps;
                float q = -xmin / s;
                q = fminf(fmaxf(q, 0.0f), maxq);
                q0 = lrintf(q);
                //q0 = q;
            }

            scales[g] = s;
            qzeros[g] = fminf(fmaxf(q0, 0.0f), maxq);
            //
        }
    }
}

__global__ void mse_build_quantization_grid(
    const float* __restrict__ X, //R*G, group_size
    float* scales, //R*G
    float* qzeros, //R*G
    const float* __restrict__ candidates, //candidates
    const int64_t  RG,
    const int64_t  P, //100
    const int64_t  group_size,
    const int bit_width,
    const float norm,
    const bool symmetric
) {

    int bid = blockIdx.x;
    int wpb = blockDim.x / 32; // 4
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wib = tid / 32; //warps in block
    int wid = (int64_t)bid * wpb + wib;

    float maxq = float((1 << bit_width) - 1);

    for (int64_t g = wid; g < (int64_t)(RG); g += (int64_t)(gridDim.x * wpb)) {
        float sg = scales[g];
        float q0g = qzeros[g];
        float xmin = 0.0;
        if (!symmetric) {
            xmin = -(sg*q0g);
        }
        float best_s = sg;
        float best_q = fminf(fmaxf(q0g, 0.0f), maxq);
        float best_loss = FLT_MAX;

        const float* group = X + (g*group_size);

        float vals[4];
        #pragma unroll 4
        for (int v = 0; v < 4; v++) {
            vals[v] = group[lane+(32*v)];
        }

        for (int p = 0; p < P; p += 1){
            float s = sg *  candidates[p];
            float q0 = 0.0;
            float rcp_s = 1.0f / s;
            if (symmetric) {
               q0 = q0g;//minf(fmaxf(lrintf(q0g), 0.0f), maxq);
            }
            else {
                q0 = -xmin*rcp_s;
                q0 = fminf(fmaxf(lrintf(q0), 0.0f), maxq);
                
            }
            float loss = 0.0;
            #pragma unroll 4
            for (int i = 0; i < 4; i += 1) {
                float v = vals[i];
                float q = lrintf(fmaf(v, rcp_s, q0));   // v * rcp0 + q0
                q       = fminf(fmaxf(q, 0.0f), maxq);
                float diff = fmaf(q - q0, s, -v);         // (q - q0)*s0 - v
                float e = fmaxf(fabsf(diff), 1e-20f);
                float lg = __logf(e);
                float val = __expf(lg * norm);
                loss += val;
            } 
            loss = warpReduceSum(loss);
            if (lane == 0) {
                if (loss < best_loss) {
                    best_loss = loss;
                    best_s = s;
                    best_q = q0;
                }
            }
        }
        if (lane == 0) {
            scales[g] = best_s;
            qzeros[g] = best_q;
        }
    }  
}


std::tuple<torch::Tensor, torch::Tensor> build_quantization_meta_cuda(
    torch::Tensor X, //R*G, 128
    int64_t bit_width,
    bool symmetric
) {

    auto device = X.device();
    const int64_t RG = X.size(0);
    const int64_t group_size = X.size(1); //must be 128...
    auto scales = torch::empty(
        {RG},
        torch::TensorOptions().dtype(torch::kFloat).device(device)
    );

    auto qzeros = torch::empty(
        {RG},
        torch::TensorOptions().dtype(torch::kFloat).device(device)
    );

    const int threads = 128;                // 4 warps/block (matches wpb = blockDim/32)
    const int wpb = threads / 32;           // 4

    // Each block covers wpb groups at a time (one per warp)
    int blocks = (int)((RG + wpb - 1) / wpb);

    // Optional clamp to avoid absurd grids (you can tune this)
    //blocks = clamp_int(blocks, 1, 65535);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();


    build_quantization_grid<<<blocks, threads, 0, stream>>>(
        X.data_ptr<float>(), 
        scales.data_ptr<float>(), 
        qzeros.data_ptr<float>(), 
        RG, group_size, 
        bit_width, symmetric);

    return std::make_tuple(scales, qzeros);
}


std::tuple<torch::Tensor, torch::Tensor> mse_quantization_grid_cuda(
    torch::Tensor X, //R*G, 128
    torch::Tensor scales,
    torch::Tensor qzeros,
    torch::Tensor candidates,
    float norm,
    int64_t bit_width,
    bool symmetric
) {

    auto device = X.device();
    const int64_t RG = X.size(0);
    const int64_t group_size = X.size(1); //must be 128...
    const int64_t P = candidates.size(0);
    

    const int threads = 128;                // 4 warps/block (matches wpb = blockDim/32)
    const int wpb = threads / 32;           // 4

    // Each block covers wpb groups at a time (one per warp)
    int blocks = (int)((RG + wpb - 1) / wpb);

    // Optional clamp to avoid absurd grids (you can tune this)
    //blocks = clamp_int(blocks, 1, 65535);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    mse_build_quantization_grid<<<blocks, threads, 0, stream>>>(
        X.data_ptr<float>(), 
        scales.data_ptr<float>(), 
        qzeros.data_ptr<float>(), 
        candidates.data_ptr<float>(), 
        RG, P, group_size, 
        bit_width, norm, symmetric);

    return std::make_tuple(scales, qzeros);
}

