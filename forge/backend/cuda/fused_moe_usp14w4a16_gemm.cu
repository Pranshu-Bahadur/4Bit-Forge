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


__device__ __forceinline__ uint64_t shfl_u64(uint64_t v, int src_lane, unsigned mask=0xFFFFFFFFu) {
    uint32_t lo = (uint32_t)(v & 0xFFFFFFFFull);
    uint32_t hi = (uint32_t)(v >> 32);
    lo = __shfl_sync(mask, lo, src_lane);
    hi = __shfl_sync(mask, hi, src_lane);
    return (uint64_t)lo | ((uint64_t)hi << 32);
}

__device__ __forceinline__ ulonglong2 shfl_u64x2(ulonglong2 v, int src_lane, unsigned mask=0xFFFFFFFFu) {
    v.x = shfl_u64(v.x, src_lane, mask);
    v.y = shfl_u64(v.y, src_lane, mask);
    return v;
}


__device__ __forceinline__ void decode(
    uint64_t u64, 
    int chunk_i,
    __nv_bfloat16& val_bf16,
    uint32_t& idx2,
    uint16_t& scale_bits
) {
        const uint32_t qw32  = (uint32_t)(u64 & 0xFFFFFFFFull);
        const uint32_t hi32  = (uint32_t)(u64 >> 32);
        const uint16_t idx16 = (uint16_t)(hi32 & 0xFFFFu);

        const uint32_t q4 = (qw32 >> (4 * chunk_i)) & 0xFu;
        idx2      = (idx16 >> (2 * chunk_i)) & 0x3u;
        scale_bits = (uint16_t)(hi32 >> 16);

        int w = (int)q4 - 8;
        val_bf16 = __float2bfloat16_rn((float)w);
};

__device__ __forceinline__ void stage_path(
    const ulonglong2* __restrict__ W,
    const int64_t curr_t, // 0,...,3
    const int64_t src_t, // 0 (f=0), 2 (f=1)
    const int64_t g2,
    const int64_t uid,
    const int64_t E,
    const int64_t oc_base,
    const int64_t R,
    const groupID

) {

    ulonglong2 qwTop = make_ulonglong2(0, 0);
    ulonglong2 qwBot = make_ulonglong2(0, 0);


    if (curr_t==src_t) {
        qwTop = W[uid * E + g2*R + oc_base + groupID];
    }

    if (curr_t==(src_t + 1)) {
        qwBot = W[uid * E + g2*R + oc_base + groupID + 8];
    }
    
    unsigned mask = __activemask();
    qwTop = shfl_u64x2(qwTop, (groupID << 2) + src_t, mask);
    qwBot = shfl_u64x2(qwBot, (groupID << 2) + (src_t + 1), mask);


    __nv_bfloat16 top_h0_lo, top_h0_hi, top_h1_lo, top_h1_hi;
    __nv_bfloat16 bot_h0_lo, bot_h0_hi, bot_h1_lo, bot_h1_hi;

    uint32_t idx2_top_h0_lo, idx2_top_h0_hi, idx2_top_h1_lo, idx2_top_h1_hi;
    uint32_t idx2_bot_h0_lo, idx2_bot_h0_hi, idx2_bot_h1_lo, idx2_bot_h1_hi;

    uint16_t sc_top_h0, sc_top_h1, sc_bot_h0, sc_bot_h1;


    const int i_lo = curr_t;      // 0..3
    const int i_hi = curr_t + 4;  // 4..7

    decode(qwTop.x, i_lo, top_h0_lo, idx2_top_h0_lo, sc_top_h0);
    decode(qwTop.x, i_hi, top_h0_hi, idx2_top_h0_hi, sc_top_h0);
    decode(qwTop.y, i_lo, top_h1_lo, idx2_top_h1_lo, sc_top_h1);
    decode(qwTop.y, i_hi, top_h1_hi, idx2_top_h1_hi, sc_top_h1);

    decode(qwBot.x, i_lo, bot_h0_lo, idx2_bot_h0_lo, sc_bot_h0);
    decode(qwBot.x, i_hi, bot_h0_hi, idx2_bot_h0_hi, sc_bot_h0);
    decode(qwBot.y, i_lo, bot_h1_lo, idx2_bot_h1_lo, sc_bot_h1);
    decode(qwBot.y, i_hi, bot_h1_hi, idx2_bot_h1_hi, sc_bot_h1);


    //Todo gather/repack metadata + inject phantom sparsity for mma.sp

    
    

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