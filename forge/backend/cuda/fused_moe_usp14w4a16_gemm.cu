#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

//this is currently being built for A100 first because 4BF is for the gpu poor, by the gpu poor

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
            XS[c * NTOK + n] = ((m_base + n) < m_end) ? X[(m_base + n) * C + c] : __float2bfloat16_rn(0.0f);
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
    return make_ulonglong2(shfl_u64(v.x, src_lane, mask), shfl_u64(v.y, src_lane, mask));
}

__device__ __constant__ uint16_t k_bf16_m8_p7[16] = {
  // -8 .. -1
  0xC100, // -8.0
  0xC0E0, // -7.0
  0xC0C0, // -6.0
  0xC0A0, // -5.0
  0xC080, // -4.0
  0xC040, // -3.0 
  0xC000, // -2.0 
  0xBF80, // -1.0

  // 0 .. 7
  0x0000, //  0.0
  0x3F80, //  1.0
  0x4000, //  2.0
  0x4040, //  3.0
  0x4080, //  4.0
  0x40A0, //  5.0
  0x40C0, //  6.0
  0x40E0  //  7.0
};


__device__ __forceinline__ uint16_t bf16bits_from_i8_small(int8_t v) {
    // v in [-8..7]
    return (uint16_t)k_bf16_m8_p7[(int)v + 8];
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

    ushort4 sc_pack;

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


__device__ __forceinline__ void stage_load(
    const __restrict__ ulonglong2* W,
    ulonglong2& qwTop,
    ulonglong2& qwBot,
    const int curr_t, // 0,...,3
    const int src_t, // t=0 (f=0), t=2 (f=1)
    const int64_t uid,
    const int64_t g2,
    const int64_t G2,
    const int64_t R,
    const int64_t oc_base,
    const int64_t groupID
) {

    if (curr_t==src_t) {
        qwTop = W[(uid * G2 + g2) * R  + oc_base + groupID];
    }

    if (curr_t==(src_t + 1)) {
        qwBot = W[(uid * G2 + g2) * R + oc_base + groupID + 8];
    }

}

__device__ __forceinline__ void stage_decode(
    const ulonglong2 qwT,
    const ulonglong2 qwB,
    const int curr_t, // 0,...,3
    const int src_t, // t=0 (f=0), t=2 (f=1)
    const int64_t groupID,
    StageOut& out
) {    

    //__activemask(); better to use entire warp acc to nvidia programming guide
    unsigned mask = 0xFFFFFFFF; 

    const ulonglong2 qwTop = shfl_u64x2(qwT, ((int)groupID << 2) + src_t, mask);
    const ulonglong2 qwBot = shfl_u64x2(qwB, ((int)groupID << 2) + (src_t + 1), mask);

    out.sc_pack.x = (uint16_t)(qwTop.x >> 48);
    out.sc_pack.y = (uint16_t)(qwBot.x >> 48);
    out.sc_pack.z = (uint16_t)(qwTop.y >> 48);
    out.sc_pack.w = (uint16_t)(qwBot.y >> 48);

    short4 top = make_short4(0, 0, 0, 0);
    short4 bot = make_short4(0, 0, 0, 0);

    uchar4 meta_nib_top = make_uchar4(0, 0, 0, 0);
    uchar4 meta_nib_bot = make_uchar4(0, 0, 0, 0);

    const int i_lo = curr_t;      // 0..3
    const int i_hi = curr_t + 4;  // 4..7

    decode(qwTop.x, i_lo, top.x, meta_nib_top.x);
    decode(qwTop.x, i_hi, top.y, meta_nib_top.y);
    decode(qwTop.y, i_lo, top.z, meta_nib_top.z);
    decode(qwTop.y, i_hi, top.w, meta_nib_top.w);

    decode(qwBot.x, i_lo, bot.x, meta_nib_bot.x);
    decode(qwBot.x, i_hi, bot.y, meta_nib_bot.y);
    decode(qwBot.y, i_lo, bot.z, meta_nib_bot.z);
    decode(qwBot.y, i_hi, bot.w, meta_nib_bot.w);

    out.top_h0 = pack_i8x4_from_i16x2(top.x, top.y);
    out.bot_h0 = pack_i8x4_from_i16x2(bot.x, bot.y);
    out.top_h1 = pack_i8x4_from_i16x2(top.z, top.w);
    out.bot_h1 = pack_i8x4_from_i16x2(bot.z, bot.w);

    out.nib_h0_lo = pack_nib2(meta_nib_top.x, meta_nib_bot.x); // 0...3
    out.nib_h0_hi = pack_nib2(meta_nib_top.y, meta_nib_bot.y); // 4...7
    out.nib_h1_lo = pack_nib2(meta_nib_top.z, meta_nib_bot.z);
    out.nib_h1_hi = pack_nib2(meta_nib_top.w, meta_nib_bot.w);
}


__device__ __forceinline__ uint32_t park_tok(uint32_t tok, int t) {
    uint32_t meta_top = 0u, meta_bot = 0u;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t pkt = (t==i)? tok : __shfl_xor_sync(0xFFFFFFFFu, tok, (t ^ i), 4);
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

/*

__device__ __forceinline__ uint32_t park(const StageOut& out, int t) {
    const uint32_t e0_0_3 = park_tok((uint32_t)out.nib_h0_lo, t);
    const uint32_t e0_4_7 = park_tok((uint32_t)out.nib_h0_hi, t);
    const uint32_t e1_0_3 = park_tok((uint32_t)out.nib_h1_lo, t);
    const uint32_t e1_4_7 = park_tok((uint32_t)out.nib_h1_hi, t);

    // Merge low+high into one 32-bit word per half (just concatenate nibbles)
    const uint32_t e0 = (e0_0_3 & 0x0000FFFFu) | (e0_4_7 & 0xFFFF0000u);
    const uint32_t e1 = (e1_0_3 & 0x0000FFFFu) | (e1_4_7 & 0xFFFF0000u);

    // selector 0 uses t=0,1 => return e0 in both
    // selector 1 uses t=2,3 => return e1 in both
    if (t < 2) return e0;
    else       return e1;
}
    */



__device__ __forceinline__ void store_tile_swiglu(
    __nv_bfloat16* X2,
    int64_t I,
    int64_t m_base, int64_t m_end,
    int groupID, int t,
    int64_t oc_base,
    const float4& gate4,
    const float4& up4
){
    #pragma unroll
    for (int j=0;j<4;++j){
        int row = groupID + ((j>=2) ? 8 : 0);
        int tok = (t<<1) + (j&1);
        int64_t m = m_base + (int64_t)tok;
        if (m < m_end){
            int64_t col = oc_base + (int64_t)row;
            float g = 0.0f;
            float u = 0.0f;

            if (j == 0) {
                g = gate4.x;
                u = up4.x;
            }
            if (j == 1) {
                g = gate4.y;
                u = up4.y;
            }
            if (j == 2) {
                g = gate4.z;
                u = up4.z;
            }
            if (j == 3) {
                g = gate4.w;
                u = up4.w;
            }
            
            float sig = 1.0f / (1.0f + __expf(-g));
            float y = (g * sig) * u;                // silu(g)*u
            X2[m*I + col] = __float2bfloat16(y);
        }
    }
}



__device__ __forceinline__ void store(
    __nv_bfloat16* Y,
    int64_t H,
    int64_t m_base, int64_t m_end,
    int groupID, int t,
    int64_t oc_base,
    const float4& D4
){
    #pragma unroll
    for (int j=0;j<4;++j){
        int row = groupID + ((j>=2) ? 8 : 0);
        int tok = (t<<1) + (j&1);
        int64_t m = m_base + (int64_t)tok;
        if (m < m_end){
            int64_t col = oc_base + (int64_t)row;
            float d = 0.0f;

            if (j == 0) {
                d = D4.x;
            }
            if (j == 1) {
                d = D4.y;
            }
            if (j == 2) {
                d = D4.z;
            }
            if (j == 3) {
                d = D4.w;
            }
            
            Y[m*H + col] = __float2bfloat16(d);
        }
    }
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
    const void XS_ptr,
    uint4& b
) {
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(XS_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w)
        : "r"(smem)
    );
}

//trans


template<int F>
__device__ __forceinline__ void mma(const uint4 a, const uint4 b, const uint32_t e, float4& c) {
  
  const float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  if constexpr (F==0) {
    asm volatile(
      "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
      : "=f"(c.x), "=f"(c.y), "=f"(c.z), "=f"(c.w)
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
        "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w),
        "f"(z.x), "f"(z.y), "f"(z.w), "f"(z.z),
        "r"(e)
    );
  } else {
    asm volatile(
      "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x1;\n"
      : "+f"(c.x), "+f"(c.y), "+f"(c.z), "+f"(c.w)
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
        "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w),
        "f"(z.x), "f"(z.y), "f"(z.w), "f"(z.z),
        "r"(e)
    );
  }
}

//        "f"(z), "f"(z), "f"(z), "f"(z), {%12,%13,%14,%15},


// assuming C <= 7168 | C is K in nvidia notation
template <int64_t NTOK, int64_t OTILE, int64_t CTA>
__global__ void phantom_usp14_w4a16_sym_sm80_fmoe_w13AS_mm_phase(
    const ulonglong2* __restrict__ W13, //[E, G2, R] | G32=(C/32), G2 = G32/2 | R=2I | C=H
    const __nv_bfloat16* __restrict__ X, //[N, C] permuted
    __nv_bfloat16* X2, // [N, R/2] permuted
    const int64_t* __restrict__ offsets, // [E+1]
    const int32_t* __restrict__ U, // [#active expert ids]
    const int64_t N,
    const int64_t C,
    const int64_t R,
    const int64_t G2
) {
    const int64_t uid = (int64_t)U[blockIdx.y];
    const int64_t m_base = offsets[uid] + (((int64_t)(blockIdx.x)) * NTOK);
    const int64_t m_end = offsets[uid + 1];

    if (m_base >= m_end) return;
    
    const int64_t tid = (int64_t)threadIdx.x;

    extern __shared__ __nv_bfloat16 XS[];
    
    stage_XS<NTOK, CTA>(X, XS, tid, m_base, m_end, C);
    __syncthreads();
    
    const int64_t wid = tid >> 5;
    const int64_t lane = tid & 31;
    const int64_t groupID = lane >> 2;
    const int64_t t = lane & 3;
    const int64_t i_base = (((int64_t)(blockIdx.z)) * OTILE);
    
    
    float4 D1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 D3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    StageOut gate;
    uint4 gate_ah0 = make_uint4(0u, 0u, 0u, 0u);
    uint4 gate_ah1 = make_uint4(0u, 0u, 0u, 0u);
    uint4 bh0 = make_uint4(0u, 0u, 0u, 0u);
    uint4 bh1 = make_uint4(0u, 0u, 0u, 0u);
    uint32_t metadata_gate = 0u;

    StageOut up;
    uint4 up_ah0 = make_uint4(0u, 0u, 0u, 0u);
    uint4 up_ah1 = make_uint4(0u, 0u, 0u, 0u);
    uint32_t metadata_up = 0u;

    ushort4 scales_gate = make_ushort4(0u, 0u, 0u, 0u);
    ushort4 scales_up = make_ushort4(0u, 0u, 0u, 0u);

    float4 fscales_gate = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 fscales_up = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    ulonglong2 qwTop = make_ulonglong2(0u, 0u);
    ulonglong2 qwBot = make_ulonglong2(0u, 0u);

    for (int64_t phase = 0; phase < 2; ++phase) {
        
        int64_t oc_base = i_base + (phase * 64) + (wid * 16);

        D1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        D3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        stage_load(W13, qwTop, qwBot, (int)t, 0, uid, 0, G2, R, oc_base, groupID);
        stage_load(W13, qwTop, qwBot, (int)t, 2, uid, 0, G2, R, oc_base + (R/2), groupID);

        stage_decode(qwTop, qwBot, (int)t, 0, groupID, gate);
        stage_decode(qwTop, qwBot, (int)t, 2, groupID, up);

        scales_gate = gate.sc_pack;
        scales_up = up.sc_pack;

        metadata_gate = park(gate, (int)t);
        metadata_up = park(up, (int)t);

        fscales_gate.x = bf16_bits_to_f32(scales_gate.x);
        fscales_gate.y = bf16_bits_to_f32(scales_gate.y);
        fscales_gate.z = bf16_bits_to_f32(scales_gate.z);
        fscales_gate.w = bf16_bits_to_f32(scales_gate.w);

        fscales_up.x = bf16_bits_to_f32(scales_up.x);
        fscales_up.y = bf16_bits_to_f32(scales_up.y);
        fscales_up.z = bf16_bits_to_f32(scales_up.z);
        fscales_up.w = bf16_bits_to_f32(scales_up.w);

        ldsmB(&XS[(((int64_t)0 << 6) + ((int64_t)0 << 5)) * NTOK], bh0);
        ldsmB(&XS[(((int64_t)0 << 6) + ((int64_t)1 << 5)) * NTOK], bh1);

        bf16x2x2_from_i8x4(gate.top_h0, gate_ah0.x, gate_ah0.y);
        bf16x2x2_from_i8x4(gate.bot_h0, gate_ah0.z, gate_ah0.w);
        bf16x2x2_from_i8x4(gate.top_h1, gate_ah1.x, gate_ah1.y);
        bf16x2x2_from_i8x4(gate.bot_h1, gate_ah1.z, gate_ah1.w);
        bf16x2x2_from_i8x4(up.top_h0, up_ah0.x, up_ah0.y);
        bf16x2x2_from_i8x4(up.bot_h0, up_ah0.z, up_ah0.w);
        bf16x2x2_from_i8x4(up.top_h1, up_ah1.x, up_ah1.y);
        bf16x2x2_from_i8x4(up.bot_h1, up_ah1.z, up_ah1.w);
        
        for (int64_t g2 = 1; g2 < G2+1; ++g2) {

            float4 C1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 C3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            mma<0>(gate_ah0, bh0, metadata_gate, C1);

            if (g2 < G2) {
                stage_load(W13, qwTop, qwBot, (int)t, 2, uid, g2, G2, R, oc_base, groupID);
            }
            

            mma<1>(up_ah1, bh1, metadata_up, C3);

            if (g2 < G2) {
                stage_load(W13, qwTop, qwBot, (int)t, 0, uid, g2, G2, R, oc_base + (R/2), groupID);
            }

            D3.x = __fmaf_rn(C3.x, fscales_up.z, D3.x);
            D3.y = __fmaf_rn(C3.y, fscales_up.z, D3.y);
            D3.z = __fmaf_rn(C3.z, fscales_up.w, D3.z);
            D3.w = __fmaf_rn(C3.w, fscales_up.w, D3.w);

            mma<0>(up_ah0, bh0, metadata_up, C3);

            if (g2 < G2) {
                ldsmB(&XS[((g2 << 6) + ((int64_t)0 << 5)) * NTOK], bh0);
            }

            D1.x = __fmaf_rn(C1.x, fscales_gate.x, D1.x);
            D1.y = __fmaf_rn(C1.y, fscales_gate.x, D1.y);
            D1.z = __fmaf_rn(C1.z, fscales_gate.y, D1.z);
            D1.w = __fmaf_rn(C1.w, fscales_gate.y, D1.w);

            mma<1>(gate_ah1, bh1, metadata_gate, C1);

            if (g2 < G2) {
                ldsmB(&XS[((g2 << 6) + ((int64_t)1 << 5)) * NTOK], bh1);
            }

            D3.x = __fmaf_rn(C3.x, fscales_up.x, D3.x);
            D3.y = __fmaf_rn(C3.y, fscales_up.x, D3.y);
            D3.z = __fmaf_rn(C3.z, fscales_up.y, D3.z);
            D3.w = __fmaf_rn(C3.w, fscales_up.y, D3.w);
           

            D1.x = __fmaf_rn(C1.x, fscales_gate.z, D1.x);
            D1.y = __fmaf_rn(C1.y, fscales_gate.z, D1.y);
            D1.z = __fmaf_rn(C1.z, fscales_gate.w, D1.z);
            D1.w = __fmaf_rn(C1.w, fscales_gate.w, D1.w);

            if (g2 < G2) {

                stage_decode(qwTop, qwBot, (int)t, 0, groupID, up);
                stage_decode(qwTop, qwBot, (int)t, 2, groupID, gate);
                scales_gate = gate.sc_pack;

                fscales_gate.x = bf16_bits_to_f32(scales_gate.x);
                fscales_gate.y = bf16_bits_to_f32(scales_gate.y);
                fscales_gate.z = bf16_bits_to_f32(scales_gate.z);
                fscales_gate.w = bf16_bits_to_f32(scales_gate.w);
                
                metadata_gate = park(gate, (int)t);
                metadata_up = park(up, (int)t);

                scales_up = up.sc_pack;
                
                fscales_up.x = bf16_bits_to_f32(scales_up.x);
                fscales_up.y = bf16_bits_to_f32(scales_up.y);
                fscales_up.z = bf16_bits_to_f32(scales_up.z);
                fscales_up.w = bf16_bits_to_f32(scales_up.w);

                bf16x2x2_from_i8x4(gate.top_h0, gate_ah0.x, gate_ah0.y);
                bf16x2x2_from_i8x4(gate.bot_h0, gate_ah0.z, gate_ah0.w);
                bf16x2x2_from_i8x4(gate.top_h1, gate_ah1.x, gate_ah1.y);
                bf16x2x2_from_i8x4(gate.bot_h1, gate_ah1.z, gate_ah1.w);
                bf16x2x2_from_i8x4(up.top_h0, up_ah0.x, up_ah0.y);
                bf16x2x2_from_i8x4(up.bot_h0, up_ah0.z, up_ah0.w);
                bf16x2x2_from_i8x4(up.top_h1, up_ah1.x, up_ah1.y);
                bf16x2x2_from_i8x4(up.bot_h1, up_ah1.z, up_ah1.w);
            }
        }

        store_tile_swiglu(X2, R/2, m_base, m_end, (int)groupID, (int)t, oc_base, D1, D3);
    }
}


// assuming C <= 2048 | C is K in nvidia notation
//NTOK=8, OTILE=128, CTA=256 (Half the reg pressure and smem allows it)
template <int64_t NTOK, int64_t OTILE, int64_t CTA>
__global__ void phantom_usp14_w4a16_sym_sm80_fmoe_w2AS_mm(
    const ulonglong2* __restrict__ W2, //[E, G2, R] | G32=(C/32), G2 = G32/2 | R=H | C=I
    const __nv_bfloat16* __restrict__ X2, //[N, C] permuted
    __nv_bfloat16* Y, // [N, R] permuted
    const int64_t* __restrict__ offsets, // [E+1]
    const int32_t* __restrict__ U, // [#active expert ids]
    const int64_t N,
    const int64_t C,
    const int64_t R,
    const int64_t G2
) {
    const int64_t uid = (int64_t)U[blockIdx.y];
    const int64_t m_base = offsets[uid] + (((int64_t)(blockIdx.x)) * NTOK);
    const int64_t m_end = offsets[uid + 1];

    if (m_base >= m_end) return;
    
    const int64_t tid = (int64_t)threadIdx.x;

    extern __shared__ __nv_bfloat16 XS[];
    
    stage_XS<NTOK, CTA>(X2, XS, tid, m_base, m_end, C);
    __syncthreads();
    
    const int64_t wid = tid >> 5;
    const int64_t lane = tid & 31;
    const int64_t groupID = lane >> 2;
    const int64_t t = lane & 3;
    const int64_t i_base = (((int64_t)(blockIdx.z)) * OTILE);
    
    float4 D = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    StageOut out;
    uint4 out_ah0 = make_uint4(0u, 0u, 0u, 0u);
    uint4 out_ah1 = make_uint4(0u, 0u, 0u, 0u);
    uint4 bh0 = make_uint4(0u, 0u, 0u, 0u);
    uint4 bh1 = make_uint4(0u, 0u, 0u, 0u);
    uint32_t metadata_out = 0u;

    ushort4 scales_out = make_ushort4(0u, 0u, 0u, 0u);

    float4 fscales_out = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    ulonglong2 qwTop = make_ulonglong2(0u, 0u);
    ulonglong2 qwBot = make_ulonglong2(0u, 0u);
        
    int64_t oc_base = i_base + (wid * 16);

    D = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    stage_load(W2, qwTop, qwBot, (int)t, 0, uid, 0, G2, R, oc_base, groupID);

    stage_decode(qwTop, qwBot, (int)t, 0, groupID, out);

    scales_out = out.sc_pack;

    metadata_out = park(out, (int)t);

    fscales_out.x = bf16_bits_to_f32(scales_out.x);
    fscales_out.y = bf16_bits_to_f32(scales_out.y);
    fscales_out.z = bf16_bits_to_f32(scales_out.z);
    fscales_out.w = bf16_bits_to_f32(scales_out.w);


    ldsmB(&XS[(((int64_t)0 << 6) + ((int64_t)0 << 5)) * NTOK], bh0);
    ldsmB(&XS[(((int64_t)0 << 6) + ((int64_t)1 << 5)) * NTOK], bh1);

    bf16x2x2_from_i8x4(out.top_h0, out_ah0.x, out_ah0.y);
    bf16x2x2_from_i8x4(out.bot_h0, out_ah0.z, out_ah0.w);
    bf16x2x2_from_i8x4(out.top_h1, out_ah1.x, out_ah1.y);
    bf16x2x2_from_i8x4(out.bot_h1, out_ah1.z, out_ah1.w);
        
    for (int64_t g2 = 1; g2 < G2+1; ++g2) {

            float4 C1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 C2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            if (g2 < G2) {
                stage_load(W2, qwTop, qwBot, (int)t, 2, uid, g2, G2, R, oc_base, groupID);
            }

            mma<1>(out_ah1, bh1, metadata_out, C2);

             if (g2 < G2) {
                ldsmB(&XS[((g2 << 6) + ((int64_t)1 << 5)) * NTOK], bh1);
            }


            mma<0>(out_ah0, bh0, metadata_out, C1);

            if (g2 < G2) {
                ldsmB(&XS[((g2 << 6) + ((int64_t)0 << 5)) * NTOK], bh0);
            }

            D.x = __fmaf_rn(C1.x, fscales_out.x, D.x);
            D.y = __fmaf_rn(C1.y, fscales_out.x, D.y);
            D.z = __fmaf_rn(C1.z, fscales_out.y, D.z);
            D.w = __fmaf_rn(C1.w, fscales_out.y, D.w);

            D.x = __fmaf_rn(C2.x, fscales_out.z, D.x);
            D.y = __fmaf_rn(C2.y, fscales_out.z, D.y);
            D.z = __fmaf_rn(C2.z, fscales_out.w, D.z);
            D.w = __fmaf_rn(C2.w, fscales_out.w, D.w);

            if (g2 < G2) {

                stage_decode(qwTop, qwBot, (int)t, 2, groupID, out);
                
                scales_out = out.sc_pack;

                fscales_out.x = bf16_bits_to_f32(scales_out.x);
                fscales_out.y = bf16_bits_to_f32(scales_out.y);
                fscales_out.z = bf16_bits_to_f32(scales_out.z);
                fscales_out.w = bf16_bits_to_f32(scales_out.w);
                
                metadata_out = park(out, (int)t);

                bf16x2x2_from_i8x4(out.top_h0, out_ah0.x, out_ah0.y);
                bf16x2x2_from_i8x4(out.bot_h0, out_ah0.z, out_ah0.w);
                bf16x2x2_from_i8x4(out.top_h1, out_ah1.x, out_ah1.y);
                bf16x2x2_from_i8x4(out.bot_h1, out_ah1.z, out_ah1.w);
            }
    }
    store(Y, R, m_base, m_end, (int)groupID, (int)t, oc_base, D);
}


torch::Tensor usp14w4a16sym_sm80_fused_moe_w13_gemm(
    torch::Tensor W13,  //[E, G2, R] | G32=(C/32), G2 = G32/2 | R=2I | C=H
    torch::Tensor X, //[N, C] permuted along N
    torch::Tensor offsets, // [E+1]
    torch::Tensor U //[#active experts <= E]
) {

    const int64_t NTOK = 8;
    const int64_t OTILE = 128;
    const int64_t CTA = 128;

    W13 = W13.contiguous();
    const int64_t G2 = (int64_t)W13.size(1);
    const int64_t R  = (int64_t)W13.size(2);

    X = X.contiguous();
    const int64_t N = (int64_t)X.size(0);
    const int64_t C = (int64_t)X.size(1);

    U = U.contiguous();

    const int64_t num_active_E = U.size(0);

    auto X2 = torch::empty({N, (R/2)}, X.options()).contiguous();

    size_t smem_bytes = NTOK * C * ((int64_t)(2));

    cudaFuncSetAttribute(
        phantom_usp14_w4a16_sym_sm80_fmoe_w13AS_mm_phase<NTOK, OTILE, CTA>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes
    );

    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 block(CTA);
    dim3 grid((unsigned)((N + NTOK - 1)/NTOK), (unsigned)num_active_E, (unsigned)(((R/2) + OTILE - 1)/OTILE));

    auto W13_ptr = reinterpret_cast<const ulonglong2*>(W13.data_ptr<uint64_t>());

    phantom_usp14_w4a16_sym_sm80_fmoe_w13AS_mm_phase<NTOK, OTILE, CTA><<<grid, block, smem_bytes, stream>>>(
        W13_ptr,
        (const __nv_bfloat16*)X.data_ptr<torch::BFloat16>(),
        (__nv_bfloat16*)X2.data_ptr<torch::BFloat16>(),
        offsets.data_ptr<int64_t>(),
        U.data_ptr<int32_t>(),
        N, C, R, G2
    );

    return X2;
}

torch::Tensor usp14w4a16sym_sm80_fused_moe_w2_gemm(
    torch::Tensor W2,  //[E, G2, R] | G32=(C/32), G2 = G32/2 | R=H | C=I
    torch::Tensor X2, //[N, C] permuted along N
    torch::Tensor offsets, // [E+1]
    torch::Tensor U //[#active experts <= E]
) {

    const int64_t NTOK = 8;
    const int64_t OTILE = 128;
    const int64_t CTA = 256;

    W2 = W2.contiguous();
    const int64_t G2 = (int64_t)W2.size(1);
    const int64_t R  = (int64_t)W2.size(2);

    X2 = X2.contiguous();
    const int64_t N = (int64_t)X2.size(0);
    const int64_t C = (int64_t)X2.size(1);
    U = U.contiguous();

    const int64_t num_active_E = U.size(0);

    auto Y = torch::empty({N, R}, X2.options()).contiguous();

    size_t smem_bytes = NTOK * C * ((int64_t)(2));

    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 block(CTA);
    dim3 grid((unsigned)((N + NTOK - 1)/NTOK), (unsigned)num_active_E, (unsigned)((R + OTILE - 1)/OTILE));

    auto W2_ptr = reinterpret_cast<const ulonglong2*>(W2.data_ptr<uint64_t>());

    phantom_usp14_w4a16_sym_sm80_fmoe_w2AS_mm<NTOK, OTILE, CTA><<<grid, block, (size_t)smem_bytes, stream>>>(
        W2_ptr,
        (const __nv_bfloat16*)X2.data_ptr<torch::BFloat16>(),
        (__nv_bfloat16*)Y.data_ptr<torch::BFloat16>(),
        offsets.data_ptr<int64_t>(),
        U.data_ptr<int32_t>(),
        N, C, R, G2
    );

    return Y;
}