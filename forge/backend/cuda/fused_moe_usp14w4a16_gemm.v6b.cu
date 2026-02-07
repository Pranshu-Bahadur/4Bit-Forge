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



__device__ __constant__ uint16_t i4_2_bf16[16] = {
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


template <int8_t TOP>
__device__ __forceinline__ uint64_t decode(
    const uint64_t u64,
    const int t
    uint64_t& midx
) {
    const uint32_t qw32  = (uint32_t)(u64 & 0xFFFFFFFFull);
    
    uint8_t idx2 = (uint8_t)(((((uint32_t)(u64 >> 32)) & 0xFFFFu) >> (t << 1)) & 0x3u);
    
    uint32_t lo = (((uint32_t)((idx2 & 1) ? 0x0000u : 0xFFFFu)) | (((uint32_t)(((idx2 & 1) ? 0xFFFFu : 0x0000u))) << 16));

    lo &= (((uint32_t)(i4_2_bf16[(int)((qw32 >> (t << 2)) & 0xFu)])) | (((uint32_t)(i4_2_bf16[(int)((qw32 >> (t << 2)) & 0xFu)])) << 16));

    midx |= (TOP) ? ((((uint64_t)((idx2 >> 1) ? 0xE : 0x4)) & 0xE) << (t << 2)) : ((((uint64_t)((idx2 >> 1) ? 0xE : 0x4)) & 0xE) << (16 + (t << 2)));

    idx2 = (uint8_t)(((((uint32_t)(u64 >> 32)) & 0xFFFFu) >> ((t + 4) << 1)) & 0x3u);

    midx |= (TOP) ? ((((uint64_t)((idx2 >> 1) ? 0xE : 0x4)) & 0xE) << (32 + (t << 2))) : ((((uint64_t)((idx2 >> 1) ? 0xE : 0x4)) & 0xE) << (32 + 16 + (t << 2)));

    uint32_t hi = (((uint32_t)((idx2 & 1) ? 0x0000u : 0xFFFFu)) | (((uint32_t)(((idx2 & 1) ? 0xFFFFu : 0x0000u))) << 16));

    hi &= (((uint32_t)(i4_2_bf16[(int)((qw32 >> ((t + 4) << 2)) & 0xFu)])) | (((uint32_t)(i4_2_bf16[(int)((qw32 >> ((t + 4) << 2)) & 0xFu)])) << 16));

    return (uint64_t)lo | (((uint64_t)hi) << 32) ;
}

__device__ __forceinline__ ulonglong2 shfl_u64x2(
    const ulonglong2 v,
    const int curr_t,
    const int src_t
) {
    unsigned long long vx = (unsigned long long)v.x;
    unsigned long long vy = (unsigned long long)v.y;
    vx = __shfl_xor_sync(0xFFFFFFFF, vx, (curr_t ^ src_t), 4);
    vy = __shfl_xor_sync(0xFFFFFFFF, vy, (curr_t ^ src_t), 4); 
    return make_ulonglong2(
        (uint64_t)vx,
        (uint64_t)vy
    );
}



struct StageOut {
    
    uint32_t top_h0, bot_h0;
    uint32_t top_h1, bot_h1;

    ushort4 bf16bit_scales;

    uint32_t nib_h0_lo, nib_h0_hi;
    uint32_t nib_h1_lo, nib_h1_hi;
};


// should be in main body same with stage load
__device__ __forceinline__ StageOut stage_decode(
    const ulonglong2 qwT,
    const ulonglong2 qwB,
    const int curr_t, // 0,...,3
    const int src_t, // t=0 (f=0), t=2 (f=1)
    const int64_t groupID
) {    

    const ulonglong2 qwTop = shfl_u64x2(qwT, curr_t, src_t);
    const ulonglong2 qwBot = shfl_u64x2(qwB, curr_t, (src_t + 1));

    out.bf16bit_scales.x = (uint16_t)(qwTop.x >> 48);
    out.bf16bit_scales.y = (uint16_t)(qwBot.x >> 48);
    out.bf16bit_scales.z = (uint16_t)(qwTop.y >> 48);
    out.bf16bit_scales.w = (uint16_t)(qwBot.y >> 48);

    uint64_t m_e0_03_47 = 0u;
    uint64_t m_e1_03_47 = 0u;

    union {
        uint64_t u64[2];
        uint4 u4;
    } cvt;


    cvt.u64[0] = decode<1>(qwTop.x, curr_t, m_e0_03_47);
    cvt.u64[1] = decode<0>(qwBot.x, curr_t, m_e0_03_47);

    uint4 frag_a_h0 = cvt.u4;

    cvt.u64[0] = decode<1>(qwTop.y, curr_t, m_e1_03_47);
    cvt.u64[1] = decode<0>(qwBot.y, curr_t, m_e1_03_47);

    uint4 frag_a_h1 = cvt.u4;

    cvt.u64[0] = m_e0_03_47;
    cvt.u64[1] = m_e1_03_47;

    uint4 metadata = cvt.u4;

}


__device__ __forceinline__ void ldsm(
    const __nv_bfloat16* __restrict__ XS_ptr,
    uint4& b
) {

    uint32_t* smem_ptr = static_cast<uint32_t*>(__cvta_generic_to_shared(XS_ptr));

    //uint32_t* b = reinterpret_cast<uint32_t*>(&frag_b);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w)
        : "r"(smem_ptr)
    );

}



// assuming C <= 7168
template <int64_t C, int64_t NTOK, int64_t OTILE>
__global__ void phantom_usp14_w4a16_sym_sm80_fmoe_w13AS_mm_phase(
    const ulonglong2* __restrict__ W13, //[E, G2, R, 2] | G32=(C/32), G2 = G32/2 | R=2I | C=H
    const __nv_bfloat16* __restrict__ X, //[N, C] permuted
    __nv_bfloat16* X2, // [N, R/2] permuted
    const int64_t* __restrict__ offsets, // [E+1]
    const int32_t* __restrict__ U, // [#active expert ids]
    const int64_t N,
    const int64_t R,
    const int64_t G2
) {
    const int64_t uid = (int64_t)U[blockIdx.y];

    const int64_t m_base = offsets[uid] + (((int64_t)(blockIdx.x)) * NTOK);
    const int64_t m_end = offsets[uid + 1];

    if (m_base >= m_end) return;
    

    extern __shared__ __nv_bfloat16 XS13[C*NTOK];
    
    for (int64_t c = ((int64_t)threadIdx.x); c < C; c += ((int64_t)blockDim.x)) {

        #pragma unroll NTOK
        for (int64_t n = 0; n < NTOK; ++n) {
            
            XS13[(c * NTOK) + n] = ((m_base + n) < m_end) ? X[((m_base + n) * C) + c] : __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    const int64_t i_base = (int64_t)(((int64_t)(blockIdx.z)) * OTILE) + (int64_t)(((threadIdx.x) >> 5) << 4);

    const int lane = (threadIdx.x) & 31;
    const int groupID = lane >> 2;
    const int t = lane & 3;

    ulonglong2 qwTop, qwBot;
    float4 D1, D3;

    uint4 bh0, bh1;

    for (int phase = 0; phase < 2; ++phase) {

        int64_t oc_base = i_base + (int64_t)(phase << 6);

        switch (t) {
            case 0: {
                qwTop = W13[(((uid * G2) + 0) * R) + oc_base + ((int64_t)groupID)];
                break;
            }

            case 1: {
                qwBot = W13[(((uid * G2) + 0) * R) + oc_base + ((int64_t)(groupID + 8))];
                break;
            }

            case 2: {
                qwTop = W13[(((uid * G2) + 0) * R) + (oc_base + (R >> 1)) + ((int64_t)groupID)];
                break;
            }

            case 3: {
                qwBot = W13[(((uid * G2) + 0) * R) + (oc_base + (R >> 1)) + ((int64_t)(groupID + 8))];
                break;
            }
        }

        float4 scales_gate, scales_up;

        ulonglong2 qw = shfl_u64x2(qwTop, t, 0);

        scales_gate.x = bf16_bits_to_f32((uint16_t)(qw.x >> 48));
        scales_gate.z = bf16_bits_to_f32((uint16_t)(qw.y >> 48));

        uint4 frag_a_h0_gate, frag_a_h1_gate, m_03_47_gate;
        ulonglong2 m_03_47 = make_ulonglong2(0u, 0u);

        asm volatile(
            "mov.b64 {%0, %1}, %4;\n\t"
            "mov.b64 {%2, %3}, %5;"
            : "=r"(frag_a_h0_gate.x), "=r"(frag_a_h0_gate.y), "=r"(frag_a_h1_gate.x), "=r"(frag_a_h1_gate.y) 
            : "l"(decode<1>(qw.x, t, m_03_47.x)), 
              "l"(decode<1>(qw.y, t, m_03_47.y))
        );

        qw = shfl_u64x2(qwBot, t, 1);

        scales_gate.y = bf16_bits_to_f32((uint16_t)(qw.x >> 48));
        scales_gate.w = bf16_bits_to_f32((uint16_t)(qw.y >> 48));

        asm volatile(
            "mov.b64 {%0, %1}, %4;\n\t"
            "mov.b64 {%2, %3}, %5;"
            : "=r"(frag_a_h0_gate.z), "=r"(frag_a_h0_gate.w), "=r"(frag_a_h1_gate.z), "=r"(frag_a_h1_gate.w) 
            : "l"(decode<0>(qw.x, t, m_03_47.x)), 
              "l"(decode<0>(qw.y, t, m_03_47.y))
        );

        asm volatile(
            "mov.b64 {%0, %1}, %4;\n\t"
            "mov.b64 {%2, %3}, %5;"
            : "=r"(m_03_47_gate.x), "=r"(m_03_47_gate.y), "=r"(m_03_47_gate.z), "=r"(m_03_47_gate.w) 
            : "l"(m_03_47.x), 
              "l"(m_03_47.y)
        );


        m_03_47 = make_ulonglong2(0u, 0u);


        qw = shfl_u64x2(qwTop, t, 2);


        scales_up.x = bf16_bits_to_f32((uint16_t)(qw.x >> 48));
        scales_up.z = bf16_bits_to_f32((uint16_t)(qw.y >> 48));

        uint4 frag_a_h0_up, frag_a_h1_up, m_03_47_up;

        asm volatile(
            "mov.b64 {%0, %1}, %4;\n\t"
            "mov.b64 {%2, %3}, %5;"
            : "=r"(frag_a_h0_up.x), "=r"(frag_a_h0_up.y), "=r"(frag_a_h1_up.x), "=r"(frag_a_h1_up.y) 
            : "l"(decode<1>(qw.x, t, m_03_47.x)), 
              "l"(decode<1>(qw.y, t, m_03_47.y))
        );


        qw = shfl_u64x2(qwBot, t, 3);

        scales_up.y = bf16_bits_to_f32((uint16_t)(qw.x >> 48));
        scales_up.w = bf16_bits_to_f32((uint16_t)(qw.y >> 48));

        asm volatile(
            "mov.b64 {%0, %1}, %4;\n\t"
            "mov.b64 {%2, %3}, %5;"
            : "=r"(frag_a_h0_up.z), "=r"(frag_a_h0_up.w), "=r"(frag_a_h1_up.z), "=r"(frag_a_h1_up.w) 
            : "l"(decode<0>(qw.x, t, m_03_47.x)), 
              "l"(decode<0>(qw.y, t, m_03_47.y))
        );

        asm volatile(
            "mov.b64 {%0, %1}, %4;\n\t"
            "mov.b64 {%2, %3}, %5;"
            : "=r"(m_03_47_up.x), "=r"(m_03_47_up.y), "=r"(m_03_47_up.z), "=r"(m_03_47_up.w) 
            : "l"(m_03_47.x), 
              "l"(m_03_47.y)
        );


        uint4 metadata = make_uint4(0u, 0u, 0u, 0u); //x, y -> (gate, up) h0; z, w -> (gate, up) h1

        switch (t) {
            case 0: {

                #pragma unroll 4
                for (int i = 0; i < 4; ++i) {
                    metadata.x |= __shfl_xor_sync(0xFFFFFFFFu, m_03_47_gate.x, (t ^ i), 4);
                    metadata.y |= __shfl_xor_sync(0xFFFFFFFFu, m_03_47_up.x, (t ^ i), 4);
                }
                break;
            }
            case 1: {

                #pragma unroll 4
                for (int i = 0; i < 4; ++i) {
                    metadata.x |= __shfl_xor_sync(0xFFFFFFFFu, m_03_47_gate.y, (t ^ i), 4);
                    metadata.y |= __shfl_xor_sync(0xFFFFFFFFu, m_03_47_up.y, (t ^ i), 4);
                }
                break;
            }
            case 2: {

                #pragma unroll 4
                for (int i = 0; i < 4; ++i) {
                    metadata.z |= __shfl_xor_sync(0xFFFFFFFFu, m_03_47_gate.z, (t ^ i), 4);
                    metadata.w |= __shfl_xor_sync(0xFFFFFFFFu, m_03_47_up.z, (t ^ i), 4);
                }
                break;
            }
            case 3: {

                #pragma unroll 4
                for (int i = 0; i < 4; ++i) {
                    metadata.z |= __shfl_xor_sync(0xFFFFFFFFu, m_03_47_gate.w, (t ^ i), 4);
                    metadata.w |= __shfl_xor_sync(0xFFFFFFFFu, m_03_47_up.w, (t ^ i), 4);
                }
                break;
            }
        }


        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(bh0.x), "=r"(bh0.y), "=r"(bh0.z), "=r"(bh0.w)
            : "l"(__cvta_generic_to_shared(&XS13[(((int64_t)0 << 6) + ((int64_t)0 << 5)) * NTOK]))
        );


        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(bh1.x), "=r"(bh1.y), "=r"(bh1.z), "=r"(bh1.w)
            : "l"(__cvta_generic_to_shared(&XS13[(((int64_t)0 << 6) + ((int64_t)1 << 5)) * NTOK]))
        );


        for (int64_t g2 = 1; g2 <= G2; ++g2) {

            float4 C1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 C3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float z = 0.0f;


            
        }






    }
}







