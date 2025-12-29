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


// (GPTQ F2B OG)
__global__ void gptq_f2b_intrablock_kernel(
    float* __restrict__ W, // {C, R} mutated W
    float* __restrict__ U, //Unorm
    uint8_t* __restrict__ qweight, // {C, R} qweight
    const float* __restrict__ scales, // {C, R} scales
    const float* __restrict__ qzeros, // {C, R} qzeros
    float* __restrict__  Eblk, // {B, R} error
    uint8_t bits,
    int64_t R, int64_t C, int64_t B, 
    int64_t start 
) {

    int tid = threadIdx.x;
    int rid = (int64_t)blockIdx.x * blockDim.x + tid;

    __shared__ float smem[32*32]; //Note block_size B = 32

    float eps  = 1e-12f;

    for (int idx = tid; idx < B * B; idx += blockDim.x) {
        int i = idx / (int)B;
        int k = idx - i * (int)B;

        float v = 0.0f;
        if (k > i) {
            v = U[(start + i) * C + start + k];
        }
        smem[i * B + k] = v;
    }
    __syncthreads();

    if (rid >= R) return;

    float x[32];

    for (int i = 0; i < B; ++i) {
        x[i] = W[(start + i) * R + rid];
    }

    const int maxq_i = (1 << bits) - 1;

    for (int t = 0; t < B; ++t) {
        int cid = start + t;

        float s = scales[(cid * R) + rid];
        float inv_s = 1/(s + eps);
        float q0 = qzeros[(cid * R) + rid];

        float error, deq;
        uint8_t qb;

        quantize_scalar(x[t], inv_s, s, q0, maxq_i, error, qb, deq);

        qweight[(cid * R) + rid] = qb;
        W[(cid * R) + rid]       = deq;
        Eblk[(t * R) + rid]      = error;

        for (int k = t+1; k < B; ++k) {
            float alpha = smem[(t * B) + k];
            x[k] = __fmaf_rn(-alpha, error, x[k]);
        }
    }
}


__global__ void pack_U_cross_T(
    float* __restrict__ U_packed,   // [C_tail, ld]
    const float* __restrict__ U,    // [C, C]
    int C, int B, int ld,
    int start, int end
) {
    int k        = blockIdx.x * blockDim.x + threadIdx.x;
    int tail_row = blockIdx.y * blockDim.y + threadIdx.y;
    int C_tail   = C - end;
    if (k >= B || tail_row >= C_tail) return;

    float a = U[(start + k) * C + (end + tail_row)];
    U_packed[tail_row * ld + k] = a;   
}




torch::Tensor gptq_solver_cuda(
    torch::Tensor W,       // [C, R]
    torch::Tensor U,  // [C, C]
    torch::Tensor scales,  //{C, R}
    torch::Tensor qzeros, //{C, R}
    int64_t bits
) {

    W      = W.contiguous();
    const int64_t C = W.size(0);
    const int64_t R = W.size(1);

    U = U.contiguous();
    scales = scales.contiguous();
    qzeros = qzeros.contiguous();

    auto qweight     = torch::empty({C, R}, torch::TensorOptions().dtype(torch::kUInt8).device(W.device()));

    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    int64_t block_size = 32;

    auto Eblk = torch::empty({block_size, R}, torch::TensorOptions().dtype(at::kFloat).device(W.device()));
    auto U_tmp = torch::empty({C, block_size},
        torch::TensorOptions().dtype(at::kFloat).device(W.device()));
    const int grid = (static_cast<int>(R) + threads - 1) / threads;

    for (int64_t block_start = 0; block_start < C; block_start += block_size) {

        const int64_t block_end = std::min(block_start + block_size, C);
        const int64_t B_long    = block_end - block_start;
        const int B             = static_cast<int>(B_long);
        const int N             = static_cast<int>(R);

        gptq_f2b_intrablock_kernel<<<grid, threads, 0, stream>>>(
            W.data_ptr<float>(),
            U.data_ptr<float>(),
            qweight.data_ptr<uint8_t>(),
            scales.data_ptr<float>(),
            qzeros.data_ptr<float>(),
            Eblk.data_ptr<float>(),
            (uint8_t)bits,
            (int64_t)R, 
            (int64_t)C,
            B_long,
            (int64_t) block_start
        );
        if (block_end < C) {
            int C_tail = (int)C - (int)block_end;

            const int tx = 16, ty = 16;
            dim3 blk(tx, ty);
            dim3 grd((B + tx - 1) / tx, (C_tail + ty - 1) / ty);

            pack_U_cross_T<<<grd, blk, 0, stream>>>(
                U_tmp.data_ptr<float>(),   // big buffer [C, block_size]
                U.data_ptr<float>(),
                (int)C, (int)B,
                (int)block_start, (int)block_end
            );

            auto U_view = U_tmp.narrow(0, 0, C_tail).narrow(1, 0, B_long); // [C_tail, B]
            auto W_tail = W.narrow(0, block_end, C_tail);                  // [C_tail, R]
            auto E_view = Eblk.narrow(0, 0, B_long);                       // [B, R]

            W_tail.addmm_(U_view, E_view, 1.0f, -1.0f);
        }
    }
    return qweight;
}