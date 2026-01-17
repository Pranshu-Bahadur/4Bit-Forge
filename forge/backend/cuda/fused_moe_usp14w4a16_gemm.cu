#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// assuming C <= 7168
template <int NTILE, int OTILE, int TPB>
__global__ void phantom_usp14_w4a16_sym_sm80_fmoe_w13AS_mm_phase(
    const ulonglong2* __restrict__ W13, //[E, G2, R, 2] | G32=(C/32), G2 = G32/2
    const __nv_bfloat16* __restrict__ X, //[N, C] permuted
    const uint16_t* __restrict__ offsets, // [E+2]
    const uint8_t* _restrict__ U, // [#active expert ids]
    __nv_bfloat16* X2, // [N, R] permuted
    const int N,
    const int C,
    const int R
) {
    const uint8_t uid = U[blockIdx.y];
    const ushort2 offset = offsets[uid];

    if ((offset.x  + blockIdx.x * NTILE) < offset.y) return;

    __shared__ __nv_bfloat16 XS[];
    
    for (int c = (int)threadIdx.x; c < C; c += TPB) {
        #pragma unroll NTILE
        for (int n = 0; n < NTILE; ++n) {
            XS[c * NTILE + n] = ((offset.x + blockIdx.x * NTILE + n) < offset.y) ? X[(offset.x + blockIdx.x * NTILE + n) * C + c] : 0.0f;
        }
    }
    __syncthreads();
    
    const int lane = threadIdx.x & 31;


}