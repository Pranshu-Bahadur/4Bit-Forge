#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>


__device__ __forceinline__ float bf16_bits_to_f32(uint16_t bits) {
    union { __nv_bfloat16 b; uint16_t u; } cvt;
    cvt.u = bits;
    return __bfloat162float(cvt.b);
}

template <int NTOK, int CTA>
__device__ __forceinline__ void stage_XS(
    const __nv_bfloat16* __restrict__ X,
    __nv_bfloat16* XS,
    const int64_t tid,
    const int64_t m_base,
    const int64_t m_end,
    const int64_t C
) {
    for (int64_t c = tid; c < C; c += CTA) {
        #pragma unroll NTOK
        for (int64_t n = 0; n < NTOK; ++n) {
            XS[c * NTOK + n] = ((m_base + n) < m_end) ? X[(m_base + n) * C + c] : (__nv_bfloat16)(0.0f);
        }
    }
}

template <int8_t SIZE>
__device__ __forceinline__ void zero(
    float* Cmatrix,
) {
    #pragma unroll SIZE
    for (int8_t i = 0; i < SIZE; ++i) {
        Cmatrix[i] = 0.0f;
    }
}


__device__ __forceinline__ void readW_broadcast(
    const ulonglong2 W,
    const int64_t t,
    const int64_t g2,
    const int64_t uid,
    const int64_t E,
    const int64_t oc_base,
    const int64_t R,
    const groupID

) {

    ulonglong2 qwTop = make_ulonglong2(0, 0);
    ulonglong2 qwBot = make_ulonglong2(0, 0);


    if (t==0) {
        qwTop = W[uid * E + g2*R + oc_base + groupID];
    }

    if (t==1) {
        qwBot = W[uid * E + g2*R + oc_base + groupID + 8];
    }
    
    qwTop = __shfl_sync(0xFFFFFFFFu, qwTop, groupID << 2);
    qwBot = __shfl_sync(0xFFFFFFFFu, qwBot, (groupID << 2) + 1);

    //@TODO continue from here

}



// assuming C <= 7168
template <int64_t NTOK, int64_t OTILE, int64_t CTA>
__global__ void phantom_usp14_w4a16_sym_sm80_fmoe_w13AS_mm_phase(
    const ulonglong2* __restrict__ W13, //[E, G2, R] | G32=(C/32), G2 = G32/2
    const __nv_bfloat16* __restrict__ X, //[N, C] permuted
    const int64_t* __restrict__ offsets, // [E+1]
    const uint8_t* _restrict__ U, // [#active expert ids]
    __nv_bfloat16* X2, // [N, R] permuted
    const int64_t N,
    const int64_t C,
    const int64_t R,
    const int64_t G2
) {
    const int64_t uid = U[blockIdx.y];
    const int64_t m_base = offsets[uid] + (((int64_t)(blockIdx.x)) * NTOK);
    const int64_t m_end = offsets[uid + 1];

    if (m_base < m_end) return;
    
    const int64_t tid = (int64_t)threadIdx.x;

    __shared__ __nv_bfloat16 XS[];
    
    stage_XS<NTOK, CTA>(X, XS, tid, m_base, m_end);
    __syncthreads();
    
    const int64_t wid = tid >> 5;
    const int64_t lane = tid & 31;
    const int64_t groupID = lane >> 2;
    const int64_t t = lane & 3;
    const int64_t i_base = (((int64_t)(blockIdx.z)) * OTILE);
    
    float C1[4];
    float C3[4];

    const ulonglong2 uTop01 = make_ulonglong2(0, 0);
    const ulonglong2 uBot01 = make_ulonglong2(0, 0);
    uint32_t meta03 = 0u;
    uint32_t meta47 = 0u;

    for (int64_t phase = 0; phase < 2; ++phase) {
        
        int64_t oc_base = i_base + (phase * 64) + (wid * 16);

        zero<4>(C1);
        zero<4>(C3);

        // Staging loop
        for (int64_t g2 = 0; g2 < G2; ++g2) {

        }

        
    }

    


}