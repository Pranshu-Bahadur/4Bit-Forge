#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

__global__ void maskupdate18(
    const float* W, //C, R
    const float* Uinv, //C, 
    uint32_t* M, //(ceil(C/32), R)
    int64_t R, int64_t C, int64_t B, //B=32
    int64_t start
) {
    int tid = threadIdx.x;
    int rid = (int64_t)blockIdx.x * blockDim.x + tid;
    bool active = (rid < R);

    uint32_t bitpacked_mask = 0;
    for (int i = 0; i < 4; ++i) {

        float score[8];
        for (int k = 0; k < 8; ++k) {
            if (active && (start + i*8 + k < C)){
                score[k] = fabsf(W[(start+i*8 + k)*R + rid]*Uinv[(start+i*8 + k)]);
            }
            else {
                score[k] = -1e30f;
            }
        }

        int best_idx = -1;
        float best_score = -1e30f;
        int r = 8;

        #pragma unroll
        for (int l = 0; l<4; ++l){
            float local_left = score[l];
            float local_right = score[r-1];

            if (local_right > best_score) {
                best_score = local_right;
                best_idx = r - 1;
            }
            if (local_left > best_score) {
                best_score = local_left;
                best_idx = l;
            }
            --r;
        }

        if (best_idx > -1 && active) {
            int bit_pos = (i * 8) + best_idx;
            bitpacked_mask |= (1u << bit_pos);
        }
    }

    if (active) {
        M[(start>>5) * R + rid] = bitpacked_mask;
    }
}
