#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>



//this is currently being built for A100 first because 4BF is for the gpu poor

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
            //contiguous along N is cleaner
            XS[c * NTOK + n] = ((m_base + n) < m_end) ? X[(m_base + n) * C + c] : (__nv_bfloat16)(0.0f);
        }
    }
}

template <int8_t SIZE>
__device__ __forceinline__ void zero(
    float* Cmatrix
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

__device__ __constant__ uint16_t k_bf16_m8_p7[16] = {
  /* -8 .. 7 bf16 bit patterns */
  0xC100, 0xC0E0, 0xC0C0, 0xC0A0, 0xC080, 0xC060, 0xC040, 0xC020,
  0x0000, 0x3F80, 0x4000, 0x4040, 0x4080, 0x40A0, 0x40C0, 0x40E0
};

__device__ __forceinline__ uint16_t bf16bits_from_i8_small(int8_t v) {
    // v in [-8..7]
    return k_bf16_m8_p7[(int)v + 8];
}


__device__ __forceinline__ uint32_t bf16x2_from_packed_i8pair(int16_t packed) {
    int8_t v0 = (int8_t)(packed & 0xFF);
    int8_t v1 = (int8_t)((packed >> 8) & 0xFF);

    uint16_t b0 = bf16bits_from_i8_small(v0);
    uint16_t b1 = bf16bits_from_i8_small(v1);

    return (uint32_t)b0 | ((uint32_t)b1 << 16);
}


__device__ __forceinline__ void bf16x2x2_from_i8x4(
    uint32_t i8x4,
    uint32_t& out_lo_bf16x2,
    uint32_t& out_hi_bf16x2
) {
    int16_t lo = (int16_t)(i8x4 & 0xFFFFu);
    int16_t hi = (int16_t)(i8x4 >> 16);

    out_lo_bf16x2 = bf16x2_from_packed_i8pair(lo);
    out_hi_bf16x2 = bf16x2_from_packed_i8pair(hi);
}

struct StageOut {
    
    uint32_t top_h0, bot_h0;
    uint32_t top_h1, bot_h1;

    uint64_t sc_pack;

    uint16_t nib_h0_lo, nib_h0_hi;
    uint16_t nib_h1_lo, nib_h1_hi;
};


__device__ __forceinline__ void decode(
    uint64_t u64,
    int chunk_i,
    int16_t& v01_packed,    
    uint8_t& meta_nibble
) {
    const uint32_t qw32  = (uint32_t)(u64 & 0xFFFFFFFFull);
    const uint32_t hi32  = (uint32_t)(u64 >> 32);
    const uint16_t idx16 = (uint16_t)(hi32 & 0xFFFFu);
    const uint32_t q4   = (qw32 >> (4 * chunk_i)) & 0xFu;
    const uint32_t idx2 = (idx16 >> (2 * chunk_i)) & 0x3u;

    const int8_t w = (int8_t)((int)q4 - 8);

    const uint32_t pair = idx2 >> 1;
    const uint32_t slot = idx2 & 1;

    meta_nibble = (pair == 0) ? (uint8_t)0x4 : (uint8_t)0xE;
    
    const int8_t v0 = (slot == 0) ? w : (int8_t)0;
    const int8_t v1 = (slot == 0) ? (int8_t)0 : w;

    const uint16_t u =
        (uint16_t)(uint8_t)v0 |
        ((uint16_t)(uint8_t)v1 << 8);
    v01_packed = (int16_t)u;
}


__device__ __forceinline__ uint32_t pack_i8x4_from_i16x2(int16_t lo_packed, int16_t hi_packed) {
    return ((uint32_t)(uint16_t)lo_packed) | ((uint32_t)(uint16_t)hi_packed << 16);
}


__device__ __forceinline__ uint16_t pack_nib2(
    uint8_t top,
    uint8_t bot
) {
    return ((uint16_t)(top & 0xF)) | (((uint16_t)(bot & 0xF)) << 4);

}


__device__ __forceinline__ void stage(
    const ulonglong2* __restrict__ W,
    const int curr_t, // 0,...,3
    const int src_t, // t=0 (f=0), t=2 (f=1)
    const int64_t uid,
    const int64_t g2,
    const int64_t G2,
    const int64_t R,
    const int64_t oc_base,
    const int64_t groupID,
    StageOut& out
) {

    ulonglong2 qwTop = make_ulonglong2(0, 0);
    ulonglong2 qwBot = make_ulonglong2(0, 0);

    //vec2 load vs slight divergence tradeoff
    if (curr_t==src_t) {
        qwTop = W[(uid * G2 + g2) * R  + oc_base + groupID];
    }

    if (curr_t==(src_t + 1)) {
        qwBot = W[(uid * G2 + g2) * R + oc_base + groupID + 8];
    }

    //__activemask(); better to use entire warp acc to nvidia programming guide
    unsigned mask = 0xFFFFFFFF; 

    qwTop = shfl_u64x2(qwTop, ((int)groupID << 2) + src_t, mask);
    qwBot = shfl_u64x2(qwBot, ((int)groupID << 2) + (src_t + 1), mask);


    uint16_t sc_top_h0 = (uint16_t)(qwTop.x >> 48);
    uint16_t sc_bot_h0 = (uint16_t)(qwBot.x >> 48);
    uint16_t sc_top_h1 = (uint16_t)(qwTop.y >> 48);
    uint16_t sc_bot_h1 = (uint16_t)(qwBot.y >> 48);

    out.sc_pack =
        (uint64_t)sc_top_h0 |
        ((uint64_t)sc_bot_h0 << 16) |
        ((uint64_t)sc_top_h1 << 32) |
        ((uint64_t)sc_bot_h1 << 48);


    int16_t top_h0_lo, top_h0_hi, top_h1_lo, top_h1_hi;
    int16_t bot_h0_lo, bot_h0_hi, bot_h1_lo, bot_h1_hi;

    uint8_t meta_nib_top_h0_lo, meta_nib_top_h0_hi, meta_nib_top_h1_lo, meta_nib_top_h1_hi;
    uint8_t meta_nib_bot_h0_lo, meta_nib_bot_h0_hi, meta_nib_bot_h1_lo, meta_nib_bot_h1_hi;

    const int i_lo = curr_t;      // 0..3
    const int i_hi = curr_t + 4;  // 4..7

    decode(qwTop.x, i_lo, top_h0_lo, meta_nib_top_h0_lo);
    decode(qwTop.x, i_hi, top_h0_hi, meta_nib_top_h0_hi);
    decode(qwTop.y, i_lo, top_h1_lo, meta_nib_top_h1_lo);
    decode(qwTop.y, i_hi, top_h1_hi, meta_nib_top_h1_hi);

    decode(qwBot.x, i_lo, bot_h0_lo, meta_nib_bot_h0_lo);
    decode(qwBot.x, i_hi, bot_h0_hi, meta_nib_bot_h0_hi);
    decode(qwBot.y, i_lo, bot_h1_lo, meta_nib_bot_h1_lo);
    decode(qwBot.y, i_hi, bot_h1_hi, meta_nib_bot_h1_hi);

    out.top_h0 = pack_i8x4_from_i16x2(top_h0_lo, top_h0_hi);
    out.bot_h0 = pack_i8x4_from_i16x2(bot_h0_lo, bot_h0_hi);
    out.top_h1 = pack_i8x4_from_i16x2(top_h1_lo, top_h1_hi);
    out.bot_h1 = pack_i8x4_from_i16x2(bot_h1_lo, bot_h1_hi);

    out.nib_h0_lo = pack_nib2(meta_nib_top_h0_lo, meta_nib_bot_h0_lo); // 0...3
    out.nib_h0_hi = pack_nib2(meta_nib_top_h0_hi, meta_nib_bot_h0_hi); // 4...7
    out.nib_h1_lo = pack_nib2(meta_nib_top_h1_lo, meta_nib_bot_h1_lo);
    out.nib_h1_hi = pack_nib2(meta_nib_top_h1_hi, meta_nib_bot_h1_hi);
}


__device__ __forceinline__ uint32_t park_tok(uint32_t tok, int t) {
    uint32_t meta_top = 0u, meta_bot = 0u;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t pkt = __shfl_xor_sync(0xFFFFFFFFu, tok, (t ^ i), 4);
        meta_top |= (pkt & 0xFu)        << (i << 2);
        meta_bot |= ((pkt >> 4) & 0xFu) << (i << 2);
    }
    return meta_top | (meta_bot << 16);
}

__device__ __forceinline__ uint32_t park(const StageOut& out, int t) {
    const uint32_t e0_0_3 = park_tok((uint32_t)out.nib_h0_lo, t); //0...3
    const uint32_t e0_4_7 = park_tok((uint32_t)out.nib_h0_hi, t); //4...7
    const uint32_t e1_0_3 = park_tok((uint32_t)out.nib_h1_lo, t);
    const uint32_t e1_4_7 = park_tok((uint32_t)out.nib_h1_hi, t);

    if (t == 0) return e0_0_3;
    if (t == 1) return e0_4_7;
    if (t == 2) return e1_0_3;
    if (t == 3) return e1_4_7;
    return 0u;
}


//@TODO mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 wrapper
//------------------------------------------------------------------------------------------------------
// A -> Fragments "A vector expression containing 4 .b32 regs with each reg containing 2 non-zero .bf16"
//     a0       a1         R      C'
// top_h*_lo, top_h*_hi (0...7, 0...7) 32 (8*4)
//     a2       a3
// bot_h*_lo, bot_h*_hi (8...15, 0...7) 32 (8*4)
//------------------------------------------------------------------------------------------------------
// B -> Fragments "A vector expression containing 4 .b32 regs, each containing 2 .bf16 elements from B (XS)"
//    b0                       b1                  b2                b3  
//    b+0+16t+(0, 1)      b+8+16t+(0, 1)     b+16+16t+(0, 1)   b+24+16t+(0, 1)  for +n=groupID forall (b=base)


__device__ __forceinline__ void ldsmB(
    const __nv_bfloat16* XS,
    uint4 b
) {
    
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(XS));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w)
        : "r"(smem)
    );
}

__device__ __forceinline__ void mma(
    const uint4 a,
    const uint4 b,
    const uint32_t meta_data,
    float* c,
    const int8_t h
) {

    if (h==0) {
        asm volatile(
            "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, {%16}, 0x0;\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
              "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
              "r"(meta_data)
        )
    }
    else {
        asm volatile(
            "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, {%16}, 0x1;\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
              "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
              "r"(meta_data)
        )
    }
}



// assuming C <= 7168 | C is K in nvidia notation
template <int64_t NTOK, int64_t OTILE, int64_t CTA>
__global__ void phantom_usp14_w4a16_sym_sm80_fmoe_w13AS_mm_phase(
    const ulonglong2* __restrict__ W13, //[E, G2, R] | G32=(C/32), G2 = G32/2 | R=2I | C=H
    const __nv_bfloat16* __restrict__ X, //[N, C] permuted
    __nv_bfloat16* X2, // [N, R] permuted
    const int64_t* __restrict__ offsets, // [E+1]
    const uint8_t* _restrict__ U, // [#active expert ids]
    const int64_t N,
    const int64_t C,
    const int64_t R,
    const int64_t G2
) {
    const int64_t uid = U[blockIdx.y];
    const int64_t m_base = offsets[uid] + (((int64_t)(blockIdx.x)) * NTOK);
    const int64_t m_end = offsets[uid + 1];

    if (m_base > m_end) return;
    
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
    
    StageOut gate;
    uint4 ah0 = make_uint4(0, 0, 0, 0);
    uint4 ah1 = make_uint4(0, 0, 0, 0);
    uint4 bh0 = make_uint4(0, 0, 0, 0);
    uint4 bh1 = make_uint4(0, 0, 0, 0);
    uint32_t metadata_gate = 0u;



    for (int64_t phase = 0; phase < 2; ++phase) {
        
        int64_t oc_base = i_base + (phase * 64) + (wid * 16);

        zero<4>(C1);
        zero<4>(C3);

        stage(
                W13, 
                (int)t, 0, 
                uid, 0, G2, R, oc_base, groupID,
                &gate
            );
        ldsmB(XS[((0 << 6) + ((int64_t)0 << 5)) * NTOK + groupID], bh0);
        ldsmB(XS[((0 << 6) + ((int64_t)1 << 5)) * NTOK + groupID], bh1);

        
        metadata = park(&gate, (int)t);

        for (int64_t g2 = 1; g2 < G2; ++g2) {

            bf16x2x2_from_i8x4(gate.top_h0, ah0.x, ah0.y);
            bf16x2x2_from_i8x4(gate.bot_h0, ah0.z, ah0.w);
            bf16x2x2_from_i8x4(gate.top_h1, ah1.x, ah1.y);
            bf16x2x2_from_i8x4(gate.bot_h1, ah1.z, ah1.w);

            mma(ah0, bh0, metadata_gate, C1, 0);

            stage(
                W13, 
                (int)t, 0, 
                uid, g2, G2, R, oc_base, groupID,
                &gate
            );

            mma(ah1, bh1, metadata_gate, C1, 1);

            metadata_gate = park(&gate, (int)t);


            ldsmB(XS[((g2 << 6) + ((int64_t)0 << 5)) * NTOK + groupID], bh0);
            ldsmB(XS[((g2 << 6) + ((int64_t)1 << 5)) * NTOK + groupID], bh1);


        }

        
    }

    


}