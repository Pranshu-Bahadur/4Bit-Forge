#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

////////////////////////////////////////////////////////////////////////////////
// bf16 bitcast helpers
////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ float bf16_bits_to_f32(uint16_t bits) {
    union { __nv_bfloat16 b; uint16_t u; } cvt;
    cvt.u = bits;
    return __bfloat162float(cvt.b);
}

static inline int64_t ceil_div_i64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

////////////////////////////////////////////////////////////////////////////////
// Why we are NOT using (dense) WGMMA here (math + systems rationale):
//
// Our weight format is unstructured 1:4 within each 4-wide group:
//  - Each 32-wide half has 8 groups of 4 -> only 8 nonzeros per half (25% density).
//  - Per half: we do exactly 8 MACs (per output r, per token n) rather than 32.
//
// If we "densify" into a dense bf16 tile to feed WGMMA, we must materialize all 32
// values per half (and apply scale), then run dense MMA on those 32 positions.
// That forces ~4x more multiply-add work versus the true sparse math:
//   sparse: 8 MACs/half      (true compute)
//   dense : 32 MACs/half     (wasted 75% MACs)
//
// On Hopper H100, sparse tensorcore acceleration is for *structured* 2:4 patterns.
// Unstructured 1:4 does NOT hit those sparse-TC fast paths, so dense WGMMA would
// be paying 4x compute + densify overhead + warpgroup coordination.
//
// Also for MoE, decode frequently has tiny per-expert M (N_e ~ 1..32), where the
// overheads of densify + warpgroup scheduling are hard to amortize. A SIMT gather+FMA
// kernel that preserves true sparsity is typically the right baseline.
// (If profiling ever shows large-N_e prefill dominates and SIMT is bandwidth-bound,
// *then* consider a second path — but keep the sparse-math baseline.)
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// 2-loop staged-X kernel (templated NTILE): stage X tile [NTILE, C] once, then compute.
// Shared memory required: NTILE * C * 2 bytes (bf16).
////////////////////////////////////////////////////////////////////////////////
template<int NTILE>
__global__ void unstructured_sparse14_int4symq_gemm_stageXS3(
    const ulonglong2* __restrict__ Wpair,   // [G2, R]  (each is two u64 for 64 channels)
    const __nv_bfloat16* __restrict__ X,    // [N, C] (N and C may be padded)
    __nv_bfloat16* __restrict__ Y,          // [N, R] (N may be padded)
    int64_t N,
    int64_t R,
    int64_t C,
    int64_t G2
) {
    // NTILE-row tile
    const int64_t nid_base = (int64_t)blockIdx.x * (int64_t)NTILE;

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;

    const int64_t rid = (int64_t)blockIdx.y * (int64_t)blockDim.x + (int64_t)tid;
    const bool valid_r = (rid < R);

    // Shared staging buffer: [NTILE, C]
    extern __shared__ __nv_bfloat16 Xs[];

    // -------- 1) Stage X for NTILE rows (single barrier; no sync in hot loop) --------
    // X is already padded with zeros in the launcher, so we can stage without row/col bounds checks
    // (N and C passed here are the padded sizes).
    for (int64_t t = (int64_t)tid; t < (int64_t)NTILE * C; t += (int64_t)blockDim.x) {
        Xs[t] = X[nid_base * C + t];
    }
    __syncthreads();

    // -------- 2) Compute: loop over g2, read X from shared, never sync again --------
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int64_t g2id = 0; g2id < G2; ++g2id) {
            ulonglong2 pkt = make_ulonglong2(0, 0);
            if (valid_r) {
                pkt = Wpair[g2id * R + rid];
            }
            // Two halves: 0..31 and 32..63
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const uint64_t u = (half == 0) ? pkt.x : pkt.y;

                const uint32_t qw32 = (uint32_t)(u & 0xFFFFFFFFull);
                const uint32_t hi   = (uint32_t)(u >> 32);

                const uint16_t idx16      = (uint16_t)(hi & 0xFFFFu);
                const uint16_t scale_bf16 = (uint16_t)(hi >> 16);
                const float scale         = bf16_bits_to_f32(scale_bf16);

                const int64_t base_c = (g2id << 6) + ((int64_t)half << 5);  // 64*g2 + 32*half

                // Each lane loads its element for this 32-wide half; then shfl selects within warp.
                const float x0 = __bfloat162float(Xs[0 * C + base_c + lane]);
                const float x1 = (NTILE >= 2) ? __bfloat162float(Xs[1 * C + base_c + lane]) : 0.0f;
                const float x2 = (NTILE >= 3) ? __bfloat162float(Xs[2 * C + base_c + lane]) : 0.0f;
                const float x3 = (NTILE >= 4) ? __bfloat162float(Xs[3 * C + base_c + lane]) : 0.0f;

                float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint32_t sparse2 = (idx16 >> (2 * i)) & 0x3u;          // 0..3
                    const uint32_t ch      = ((uint32_t)i << 2) + sparse2;       // 0..31
                    const float w = (float)(((qw32 >> (4 * i)) & 0xFu) - 8);     // symmetric int4 -> [-8, 7]

                    sum0 += __shfl_sync(0xFFFFFFFFu, x0, (int)ch) * w;
                    if (NTILE >= 2) sum1 += __shfl_sync(0xFFFFFFFFu, x1, (int)ch) * w;
                    if (NTILE >= 3) sum2 += __shfl_sync(0xFFFFFFFFu, x2, (int)ch) * w;
                    if (NTILE >= 4) sum3 += __shfl_sync(0xFFFFFFFFu, x3, (int)ch) * w;
                }

                acc0 += sum0 * scale;
                if (NTILE >= 2) acc1 += sum1 * scale;
                if (NTILE >= 3) acc2 += sum2 * scale;
                if (NTILE >= 4) acc3 += sum3 * scale;
            }
    }
    

    // Stores (N is padded, so writes are always in-bounds; R is predicated via valid_r).
    if (valid_r) {
        Y[(nid_base + 0) * R + rid] = __float2bfloat16(acc0);
        if (NTILE >= 2) Y[(nid_base + 1) * R + rid] = __float2bfloat16(acc1);
        if (NTILE >= 3) Y[(nid_base + 2) * R + rid] = __float2bfloat16(acc2);
        if (NTILE >= 4) Y[(nid_base + 3) * R + rid] = __float2bfloat16(acc3);
    }
}


template<int NTILE>
__global__ void unstructured_sparse14_int4symq_gemm(
    const ulonglong2* __restrict__ Wpair,   // [G2, R]  (each is two u64 for 64 channels)
    const __nv_bfloat16* __restrict__ X,    // [N, C] (N and C may be padded)
    __nv_bfloat16* __restrict__ Y,          // [N, R] (N may be padded)
    int64_t N,
    int64_t R,
    int64_t C,
    int64_t G2
) {
    // NTILE-row tile
    const int64_t nid_base = (int64_t)blockIdx.x * (int64_t)NTILE;

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;

    const int64_t rid = (int64_t)blockIdx.y * (int64_t)blockDim.x + (int64_t)tid;
    const bool valid_r = (rid < R);

    // -------- 2) Compute: loop over g2, read X from shared, never sync again --------
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int64_t g2id = 0; g2id < G2; ++g2id) {
            ulonglong2 pkt = make_ulonglong2(0, 0);
            if (valid_r) {
                pkt = Wpair[g2id * R + rid];
            }
            // Two halves: 0..31 and 32..63

            for (int half = 0; half < 2; ++half) {
                const uint64_t u = (half == 0) ? pkt.x : pkt.y;

                const uint32_t qw32 = (uint32_t)(u & 0xFFFFFFFFull);
                const uint32_t hi   = (uint32_t)(u >> 32);

                const uint16_t idx16      = (uint16_t)(hi & 0xFFFFu);
                const uint16_t scale_bf16 = (uint16_t)(hi >> 16);
                const float scale         = bf16_bits_to_f32(scale_bf16);

                const int64_t base_c = (g2id << 6) + ((int64_t)half << 5);  // 64*g2 + 32*half

                // Each lane loads its element for this 32-wide half; then shfl selects within warp.
                const float x0 = __bfloat162float(X[(n + 0) * C + base_c + lane]);
                const float x1 = (NTILE >= 2) ? __bfloat162float(X[(n + 1) * C + base_c + lane]) : 0.0f;
                const float x2 = (NTILE >= 3) ? __bfloat162float(X[(n + 2) * C + base_c + lane]) : 0.0f;
                const float x3 = (NTILE >= 4) ? __bfloat162float(X[(n + 3) * C + base_c + lane]) : 0.0f;

                float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint32_t sparse2 = (idx16 >> (2 * i)) & 0x3u;          // 0..3
                    const uint32_t ch      = ((uint32_t)i << 2) + sparse2;       // 0..31
                    const float w = (float)(((qw32 >> (4 * i)) & 0xFu) - 8);     // symmetric int4 -> [-8, 7]

                    sum0 += __shfl_sync(0xFFFFFFFFu, x0, (int)ch) * w;
                    if (NTILE >= 2) sum1 += __shfl_sync(0xFFFFFFFFu, x1, (int)ch) * w;
                    if (NTILE >= 3) sum2 += __shfl_sync(0xFFFFFFFFu, x2, (int)ch) * w;
                    if (NTILE >= 4) sum3 += __shfl_sync(0xFFFFFFFFu, x3, (int)ch) * w;
                }

                acc0 += sum0 * scale;
                if (NTILE >= 2) acc1 += sum1 * scale;
                if (NTILE >= 3) acc2 += sum2 * scale;
                if (NTILE >= 4) acc3 += sum3 * scale;
            }
    }
    

    // Stores (N is padded, so writes are always in-bounds; R is predicated via valid_r).
    if (valid_r) {
        Y[(nid_base + 0) * R + rid] = __float2bfloat16(acc0);
        if (NTILE >= 2) Y[(nid_base + 1) * R + rid] = __float2bfloat16(acc1);
        if (NTILE >= 3) Y[(nid_base + 2) * R + rid] = __float2bfloat16(acc2);
        if (NTILE >= 4) Y[(nid_base + 3) * R + rid] = __float2bfloat16(acc3);
    }
}

static inline int pick_NTILE(int64_t N) {
    // Heuristic dispatch:
    // - decode / tiny per-expert N: keep staging small -> NTILE=1
    // - medium: NTILE=2 or 3
    // - large prefill chunks: NTILE=4 (better reuse across the block)
    if (N <= 64)   return 1;
    if (N <= 256)  return 2;
    if (N <= 768)  return 3;
    return 4;
}

template<int NTILE>
static inline void launch_stageXS(
    const ulonglong2* Wpair,
    const __nv_bfloat16* X,
    __nv_bfloat16* Y,
    int64_t N,
    int64_t R,
    int64_t C,
    int64_t G2,
    dim3 grid,
    dim3 block,
    cudaStream_t stream
) {

    // Opt-in to larger dynamic shared memory when needed/available (A100/H100).
    // If the device can't satisfy it, the launcher will downshift NTILE before calling here.
    //cudaFuncSetAttribute(
    //    (const void*)unstructured_sparse14_int4symq_gemm_stageXS3<NTILE>,
    //   cudaFuncAttributeMaxDynamicSharedMemorySize,
    //    (int)shmem_bytes
    //);

    if (N_TILE==1) {
        unstructured_sparse14_int4symq_gemm<NTILE>
        <<<grid, block, 0, stream>>>(
            Wpair, X, Y, N, R, C, G2
        );

    }
    else {
        const int64_t shmem_bytes = (int64_t)NTILE * C * (int64_t)sizeof(__nv_bfloat16);
        unstructured_sparse14_int4symq_gemm_stageXS3<NTILE>
        <<<grid, block, (size_t)shmem_bytes, stream>>>(
            Wpair, X, Y, N, R, C, G2
        );

    }
    
}

torch::Tensor moe_proj_unstructured_sparse14_int4symq_gemm(
    torch::Tensor qW2S1u64, // [G2, R, 2] | G2=ceil(C/64) | ulonglong2 | Packing format defined above
    torch::Tensor X         // [N, C] | bfloat16
) {
    TORCH_CHECK(qW2S1u64.is_cuda(), "qW2S1u64 must be CUDA");
    TORCH_CHECK(X.is_cuda(), "X must be CUDA");
    //TORCH_CHECK(qW2S1u64.scalar_type() == torch::kULong, "qW2S1u64 must be uint64");
    TORCH_CHECK(X.scalar_type() == torch::kBFloat16, "X must be bfloat16");
    TORCH_CHECK(qW2S1u64.dim() == 3 && qW2S1u64.size(2) == 2, "qW2S1u64 must be [G2, R, 2]");
    TORCH_CHECK(X.dim() == 2, "X must be [N, C]");

    qW2S1u64 = qW2S1u64.contiguous();
    const int64_t G2 = qW2S1u64.size(0);
    const int64_t R  = qW2S1u64.size(1);

    X = X.contiguous();
    const int64_t N = X.size(0);
    const int64_t C = X.size(1);

    // C must be padded to 64 because the kernel indexes [g2*64 + half*32 + lane]
    const int64_t c_remainder = C % 64;
    const int64_t pad_c = (c_remainder == 0) ? 0 : (64 - c_remainder);
    const int64_t C_padded = C + pad_c;

    // Sanity: packed G2 must match ceil_div(C,64); padding C to 64 keeps it identical.
    TORCH_CHECK(G2 == ceil_div_i64(C, 64), "qW2S1u64 G2 mismatch: got ", G2, " expected ", ceil_div_i64(C, 64));

    // Choose NTILE based on (unpadded) N, then pad N to a multiple of NTILE.
    int NTILE = pick_NTILE(N);

    // If dynamic shared would be too large for this device, downshift NTILE until it fits.
    int device = at::cuda::current_device();
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    size_t max_smem = prop.sharedMemPerBlock;

    auto smem_needed = [&](int t){ return (size_t)t * (size_t)C_padded * sizeof(__nv_bfloat16); };
    while (NTILE > 1 && smem_needed(NTILE) > max_smem) NTILE--;

    const int64_t n_remainder = N % NTILE;
    const int64_t pad_n = (n_remainder == 0) ? 0 : (NTILE - n_remainder);
    const int64_t N_padded = N + pad_n;

    // Pad X on GPU if needed (both N and C). Padding value is 0.
    if (pad_n > 0 || pad_c > 0) {
        auto Xp = torch::zeros({N_padded, C_padded}, X.options());
        // Copy original X into top-left corner.
        Xp.slice(0, 0, N).slice(1, 0, C).copy_(X);
        X = Xp.contiguous();
    } else if (C_padded != C) {
        // (Shouldn't happen because pad_c==0 here, but keep it explicit.)
        auto Xp = torch::zeros({N_padded, C_padded}, X.options());
        Xp.slice(0, 0, N).slice(1, 0, C).copy_(X);
        X = Xp.contiguous();
    }

    auto Y = torch::empty({N_padded, R}, X.options()).contiguous();

    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 block(128);
    dim3 grid(
        (unsigned)ceil_div_i64(N_padded, NTILE),               // tiles over N
        (unsigned)ceil_div_i64(R, (int64_t)block.x)            // tiles over R
    );

    auto qW_ptr = reinterpret_cast<const ulonglong2*>(qW2S1u64.data_ptr<uint64_t>());

    // Dispatch the specialized kernel.
    if (NTILE == 1) {
        launch_stageXS<1>(qW_ptr, (const __nv_bfloat16*)X.data_ptr<torch::BFloat16>(),
                          (__nv_bfloat16*)Y.data_ptr<torch::BFloat16>(),
                          N_padded, R, C_padded, G2, grid, block, stream);
    } else if (NTILE == 2) {
        launch_stageXS<2>(qW_ptr, (const __nv_bfloat16*)X.data_ptr<torch::BFloat16>(),
                          (__nv_bfloat16*)Y.data_ptr<torch::BFloat16>(),
                          N_padded, R, C_padded, G2, grid, block, stream);
    } else if (NTILE == 3) {
        launch_stageXS<3>(qW_ptr, (const __nv_bfloat16*)X.data_ptr<torch::BFloat16>(),
                          (__nv_bfloat16*)Y.data_ptr<torch::BFloat16>(),
                          N_padded, R, C_padded, G2, grid, block, stream);
    } else { // NTILE == 4
        launch_stageXS<4>(qW_ptr, (const __nv_bfloat16*)X.data_ptr<torch::BFloat16>(),
                          (__nv_bfloat16*)Y.data_ptr<torch::BFloat16>(),
                          N_padded, R, C_padded, G2, grid, block, stream);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Unpad N if needed.
    if (pad_n > 0) return Y.slice(0, 0, N);
    return Y;
}