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
#include <stdint.h>

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


static inline int64_t ceil_div_i64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

////////////////////////////////////////////////////////////////////////////////
// Decode snippet (for matmul)
////////////////////////////////////////////////////////////////////////////////
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

// Example dequant inside matmul (symmetric, uint4 with q0 = 8 for 4-bit):
// w = (q - 8) * s
__device__ __forceinline__ float dequant_sym_u4(uint32_t q4, float s) {
    return (float((int)q4) - 8.0f) * s;
}

__global__ void pack_sparsegptq14_to_u64x2(
    const uint8_t* __restrict__ qweight,   // [C, R]
    const uint32_t* __restrict__ M,        // [G32, R]
    const float* __restrict__ scales,      // [G32, R]
    ulonglong2* __restrict__ Wpair,        // [G2, R]  (two u64s per g2)
    int64_t C, int64_t R, int64_t G32
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t r   = tid % R;
    int64_t g2  = tid / R;
    int64_t G2  = (G32 + 1) >> 1;
    if (r >= R || g2 >= G2) return;

    int64_t g0 = (g2 << 1);
    int64_t g1 = g0 + 1;

    auto pack_one = [&](int64_t gid)->uint64_t {
        if (gid >= G32) return 0ull;
        uint32_t bitmask = M[gid * R + r];

        uint16_t idx16 = 0;
        uint32_t qw32  = 0;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint32_t sub = (bitmask >> (4 * i)) & 0xFu;
            if (sub == 0u) continue;
            int c = __ffs(sub) - 1; // 0..3
            idx16 |= (uint16_t)(c & 0x3) << (2 * i);

            int64_t cid = gid * 32 + i * 4 + c;
            if (cid < C) {
                uint32_t q = (uint32_t)(qweight[cid * R + r] & 0xFu);
                qw32 |= (q << (4 * i));
            }
        }

        float s = scales[gid * R + r];
        uint16_t s_bf16 = f32_to_bf16_bits(s);

        return ((uint64_t)s_bf16 << 48) | ((uint64_t)idx16 << 32) | (uint64_t)qw32;
    };

    ulonglong2 out;
    out.x = pack_one(g0);
    out.y = pack_one(g1);
    Wpair[g2 * R + r] = out;
}


torch::Tensor pack_sparsegptq14_to_u64x2_cuda(
    torch::Tensor qweight_rc,
    torch::Tensor M,
    torch::Tensor scales
) {
    
    //CHECK_DTYPE(qweight_rc, torch::kUInt8);
    //CHECK_DTYPE(M, torch::kUInt32);
    //CHECK_DTYPE(scales, torch::kFloat32);

    // We’ll transpose inside launcher as requested.
    qweight_rc = qweight_rc.contiguous();
    M          = M.contiguous();
    scales     = scales.contiguous();

    TORCH_CHECK(qweight_rc.dim() == 2, "qweight_rc must be [R, C]");
    TORCH_CHECK(M.dim() == 2,          "M must be [G32, R]");
    TORCH_CHECK(scales.dim() == 2,     "scales must be [G32, R]");

    const int64_t R = qweight_rc.size(0);
    const int64_t C = qweight_rc.size(1);

    TORCH_CHECK(M.size(1) == R, "M second dim must equal R");
    TORCH_CHECK(scales.size(1) == R, "scales second dim must equal R");

    // Transpose to [C, R] to match kernel indexing
    auto qweight_cr = qweight_rc.transpose(0, 1).contiguous(); // [C, R]
    //CHECK_CONTIG(qweight_cr);

    // Derive/validate G32
    const int64_t G32_expected = ceil_div_i64(C, 32);
    const int64_t G32 = M.size(0);
    TORCH_CHECK(G32 == G32_expected,
        "M first dim (G32) mismatch: got ", G32, " expected ", G32_expected);

    TORCH_CHECK(scales.size(0) == G32, "scales first dim must equal G32");

    const int64_t G2 = (G32 + 1) >> 1;

    // Output tensor: store as uint64 [G2, R, 2], reinterpret as ulonglong2*
    auto out_opts = torch::TensorOptions()
        .dtype(torch::kUInt64)
        .device(qweight_rc.device());

    auto Wpair_u64 = torch::empty({G2, R, 2}, out_opts).contiguous();
    //CHECK_CONTIG(Wpair_u64);

    //const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight_rc));
    auto stream = at::cuda::getDefaultCUDAStream();

    const int threads = 256;
    const int64_t total = G2 * R;
    const dim3 block(threads);
    const dim3 grid((unsigned)ceil_div_i64(total, threads));

    pack_sparsegptq14_to_u64x2<<<grid, block, 0, stream>>>(
        (const uint8_t*)qweight_cr.data_ptr<uint8_t>(),
        (const uint32_t*)M.data_ptr<uint32_t>(),
        (const float*)scales.data_ptr<float>(),
        (ulonglong2*)Wpair_u64.data_ptr<uint64_t>(),
        C, R, G32
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Wpair_u64;
}
