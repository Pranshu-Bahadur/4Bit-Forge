#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <algorithm>


__device__ __forceinline__ void quantize_scalar(
    float x, float inv_s, float s, float q0, int maxq_i,
    float& error_out, uint8_t& q_out, float& deq_out
) {
    float biased = x * inv_s + q0;
    int q = lrintf(biased);

    if (q < 0) q = 0;
    else if (q > maxq_i) q = maxq_i;

    deq_out   = __fmaf_rn(static_cast<float>(q), s, -q0 * s);
    error_out = x - deq_out;       // W_old - Q
    q_out     = static_cast<uint8_t>(q);
}

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

// (SPARSEGPTQ 1:8 F2B OG)
__global__ void sparsegptq_f2b_intrablock_kernel(
    float* __restrict__ W, 
    const float* __restrict__ U, // U/Uinv {C, C}
    const float* __restrict__ Uinv,  
    uint32_t* __restrict__ M,
    uint8_t* __restrict__ qweight, 
    const float* __restrict__ scales, 
    const float* __restrict__ qzeros, 
    float* __restrict__  Eblk, 
    uint8_t bits,
    int64_t R, int64_t C, int64_t B, 
    int64_t start 
) {
    int tid = threadIdx.x;
    int rid = (int64_t)blockIdx.x * blockDim.x + tid;
    int lane = tid & 31;
    
    unsigned mask = 0xffffffff; 

    float eps  = 1e-12f;

    float y[32];
    for (int r = 0; r < 32; ++r) {
        float v = 0.f;
        if (r < B && lane < B && r <= lane) {
            v = U[(start + r) * C + (start + lane)];
        }
        y[r] = v;
    }

 
    float x[32];
    bool active = (rid < R); 

    if (active) {
        for (int i = 0; i < B; ++i) x[i] = W[(start + i) * R + rid];
    } else {
        
        for (int i = 0; i < B; ++i) x[i] = 0.0f;
    }

    const int maxq_i = (1 << bits) - 1;
    uint32_t bitpacked_mask = 0u;

    for (int t = 0; t < B; ++t) {
        int cid = start + t;
        if (t % 8 == 0){
            float best_score = -1e30f;
            int best_idx = -1;
            float score = best_score;
            for (int k = 0; k < 8; ++k) {
                if (active && (cid + k < C)){
                    score = fabsf(x[t+k]*Uinv[cid+k]);
                }
                else {
                    score = -1e30f;
                }
                if (best_score < score && active && (cid + k < C)) {
                    best_score = score;
                    best_idx = k;
                }
            }

            if (best_idx > -1 && active) {
                int bit_pos = t + best_idx;
                    bitpacked_mask |= (1u << bit_pos);
            }
        }
        float s = 1.0f; 
        float q0 = 0.0f;
        
        if (active) {
            s = scales[(cid * R) + rid];
            q0 = qzeros[(cid * R) + rid];
        }
        
        float inv_s = 1.0f / (s + eps);
        float error = 0.f, deq = 0.f;
        uint8_t qb = 0;

        bool keep = (bitpacked_mask >> t) & 1u;

        if (keep) {
            quantize_scalar(x[t], inv_s, s, q0, maxq_i, error, qb, deq);
        }
        else {
            error = x[t];
        }


        if (active) {
            qweight[(cid * R) + rid] = qb;
            W[(cid * R) + rid]       = deq;
            Eblk[(t * R) + rid]      = error;
        }

        
        for (int k = t + 1; k < B; ++k) {
            float alpha = __shfl_sync(mask, y[t], k);
            x[k] = __fmaf_rn(-alpha, error, x[k]);
        }
    }

    if (active) {
        M[(start>>5) * R + rid] = bitpacked_mask;
    }
}