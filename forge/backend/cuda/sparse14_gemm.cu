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

// Example dequant inside matmul (symmetric, ulonglong2 with q0 = 8 for 4-bit):
// w = (q - 8) * s
__device__ __forceinline__ float dequant_sym_u4(uint32_t q4, float s) {
    return (float((int)q4) - 8.0f) * s;
}


////////////////////////////////////////////////////////////////////////////////
// Pack format (1:4 inside 32)
// packed_u64 layout:
//
// [63:48] scale_bf16_bits | 1 scale every 32 channels (Note: could also store q0*s iff every 64 channels)
// [47:32] idx16  (8 groups * 2 bits) | each "group" has 1 value
// [31: 0] qw32   (8 groups * 4 bits) | each "group" has 1 value
//
////////////////////////////////////////////////////////////////////////////////

// Unstructured 1 out of 4 Sparsity MatMul
__global__ void unstructured_sparse14_int4symq_gemm(
    const ulonglong2* __restrict__ qW2S1u64vec2, //ulong2 [G2, R, 2] R=2048/7168
    const __nv_bfloat16* __restrict__ X, // [N, C] | __nv_bfloat16
    __nv_bfloat16* Y,             // [N, R] | __nv_bfloat16
    const int64_t N,
    const int64_t R,
    const int64_t C,
    const int64_t G2 // G = ceil(C/32) | G2 = ceil(C/64)
) {
    int64_t nid = blockIdx.x * 4;
    if ((nid) >= N) return;

    int64_t tid = threadIdx.x;
    int64_t rid = (blockIdx.y * blockDim.x) + tid;
    if (rid >= R) return;
    int64_t lane = tid & 31;
    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int64_t g2id = 0; g2id < G2; ++g2id) {

        ulonglong2 packets = qW2S1u64vec2[g2id * R + rid];

        {
            uint64_t u0 = packets.x;
            uint32_t qw32 = (uint32_t)u0;
            uint32_t meta = (uint32_t)(u0 >> 32);
            uint32_t idx16 = meta & 0xFFFF;
            float scale = bf16_to_float(meta >> 16);
            int64_t cid = (g2id << 6) + 0;

            float x0 = __bfloat162float(X[(nid + 0) * C + cid + lane]);
            float x1 = __bfloat162float(X[(nid + 1) * C + cid + lane]);
            float x2 = __bfloat162float(X[(nid + 2) * C + cid + lane]);
            float x3 = __bfloat162float(X[(nid + 3) * C + cid + lane]);

            float4 local_acc4 = {0.0f, 0.0f, 0.0f, 0.0f};

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint32_t channel = (i << 2) + ((idx16 >> (i * 2)) & 0x3);
                float deqw = (float)(((qw32 >> (i * 4)) & 0xF) - 8); // symmetric only
                local_acc4.x += __shfl_sync(0xffffffff, x0, channel) * deqw;
                local_acc4.y += __shfl_sync(0xffffffff, x1, channel) * deqw;
                local_acc4.z += __shfl_sync(0xffffffff, x2, channel) * deqw;
                local_acc4.w += __shfl_sync(0xffffffff, x3, channel) * deqw;
            }
            acc.x += local_acc4.x*scale;
            acc.y += local_acc4.y*scale;
            acc.z += local_acc4.z*scale;
            acc.w += local_acc4.w*scale;
        }

        {
            uint64_t u0 = packets.y;
            uint32_t qw32 = (uint32_t)u0;
            uint32_t meta = (uint32_t)(u0 >> 32);
            uint32_t idx16 = meta & 0xFFFF;
            float scale = bf16_to_float(meta >> 16);
            int64_t cid = (g2id << 6) + 32;

            float x0 = __bfloat162float(X[(nid + 0) * C + cid + lane]);
            float x1 = __bfloat162float(X[(nid + 1) * C + cid + lane]);
            float x2 = __bfloat162float(X[(nid + 2) * C + cid + lane]);
            float x3 = __bfloat162float(X[(nid + 3) * C + cid + lane]);

            float4 local_acc4 = {0.0f, 0.0f, 0.0f, 0.0f};

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint32_t channel = (i << 2) + ((idx16 >> (i * 2)) & 0x3);
                float deqw = (float)(((qw32 >> (i * 4)) & 0xF) - 8); // symmetric only
                local_acc4.x += __shfl_sync(0xffffffff, x0, channel) * deqw;
                local_acc4.y += __shfl_sync(0xffffffff, x1, channel) * deqw;
                local_acc4.z += __shfl_sync(0xffffffff, x2, channel) * deqw;
                local_acc4.w += __shfl_sync(0xffffffff, x3, channel) * deqw;
            }
            acc.x += local_acc4.x*scale;
            acc.y += local_acc4.y*scale;
            acc.z += local_acc4.z*scale;
            acc.w += local_acc4.w*scale;
        }
    }
    Y[((nid) + 0) * R + rid] = __float2bfloat16(acc.x);
    Y[((nid) + 1) * R + rid] = __float2bfloat16(acc.y);
    Y[((nid) + 2) * R + rid] = __float2bfloat16(acc.z);
    Y[((nid) + 3) * R + rid] = __float2bfloat16(acc.w);
}

//IMPORTANT: @TODO PLS PAD N by 4, and G by 2, and C by 64
torch::Tensor moe_proj_unstructured_sparse14_int4symq_gemm(
    torch::Tensor qW2S1u64, // [G2, R, 2] | G2=ceil(C/64) | ulonglong2 | Packing format defined above
    torch::Tensor X // [N, C] | bfloat16
) {

    qW2S1u64 = qW2S1u64.contiguous();
    const int64_t G2 = qW2S1u64.size(0);
    const int64_t R = qW2S1u64.size(1); // out_features

    X = X.contiguous();
    const int64_t N = X.size(0); // Batch Size = B*Tokens??
    const int64_t C = X.size(1); // in_features / channels

    const int N_TILE = 4;
    int64_t n_remainder = N % N_TILE;
    int64_t pad_n = (n_remainder == 0) ? 0 : (N_TILE - n_remainder);

    int64_t c_remainder = C % 64;
    int64_t pad_c = (c_remainder == 0) ? 0 : (64 - c_remainder);
    
    if (pad_n > 0 || pad_c > 0) {
        namespace F = torch::nn::functional; 
        std::vector<int64_t> pad_vec = {0, pad_c, 0, pad_n}; // Last dim output first
        X = torch::constant_pad_nd(X, pad_vec, 0).contiguous(); 
    }
    int64_t N_padded = X.size(0);
    int64_t C_padded = X.size(1);

    auto Y = torch::empty({N_padded, R}, torch::TensorOptions().dtype(torch::kBFloat16).device(X.device())).contiguous();

    auto stream = at::cuda::getCurrentCUDAStream();

    int n_vec_rows = (static_cast<int>(N_padded) + 3) / 4;
    dim3 block(128);
    dim3 grid(
        n_vec_rows, 
        (static_cast<int>(R) + block.x - 1) / block.x
    );

    auto qW_ptr = reinterpret_cast<const ulonglong2*>(qW2S1u64.data_ptr<uint64_t>());
    
    unstructured_sparse14_int4symq_gemm<<<grid, block>>>(
            qW_ptr,
            X.data_ptr<__nv_bfloat16>(), 
            Y.data_ptr<__nv_bfloat16>(),
            N_padded,
            R,
            C_padded,
            G2
    );

    if (pad_n > 0) {
        return Y.slice(0, 0, N);
    }
    return Y;
}
