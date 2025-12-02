#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

#define WARP_SIZE      32
#define TILE_COLS      32
#define MAX_GROUP_SIZE 128  // supports group_size up to this

// ---------------------------------------------------------------------
// Warp-cooperative 32x32 butterfly transpose on T[32] per lane.
// Precondition (per lane = r, index = c):
//   T[c] = A[c][r]   (row = c, col = r)
// Postcondition:
//   T[c] = A[r][c]   (row = r, col = c)
// This is the same pattern used in PackBoost's _et_sample_1b_butterfly.
// ---------------------------------------------------------------------
__device__ inline void butterfly_transpose_32(int T[WARP_SIZE], unsigned mask) {
    #pragma unroll
    for (int s = 0; s < 5; ++s) {  // ofs = 1,2,4,8,16
        const int ofs = 1 << s;

        int U[WARP_SIZE];
        #pragma unroll
        for (int i = 0; i < WARP_SIZE; ++i) {
            U[i] = T[i];
        }

        #pragma unroll
        for (int i = 0; i < WARP_SIZE; ++i) {
            int partner = __shfl_xor_sync(mask, U[i ^ ofs], ofs, WARP_SIZE);
            T[i] = (((threadIdx.x ^ i) & ofs) ? partner : U[i]);
        }
    }
}

// ---------------------------------------------------------------------
// Forge AWQ W4 v2 kernel
//
// weights: [IC, OC], float16
// qweight: [IC, OC_packs], int32
// scales : [G,  OC], float16
// qzeros : [G,  OC_packs], int32
// group_size: AWQ group along IC (rows)
//
// Grid:
//   blockDim.x = 32 (1 warp/block)
//   grid.x = ceil(OC / 32)   (32-column tiles)
//   grid.y = G = ceil(IC / group_size)  (row groups)
// ---------------------------------------------------------------------
__global__ void forge_awq_v2_kernel(
    const half* __restrict__ W,       // [IC, OC]
    int32_t* __restrict__ qweight,    // [IC, OC_packs]
    half* __restrict__ scales,        // [G,  OC]
    int32_t* __restrict__ qzeros,     // [G,  OC_packs]
    int IC,
    int OC,
    int group_size)
{
    const int lane = threadIdx.x;  // 0..31
    if (blockDim.x != WARP_SIZE || lane >= WARP_SIZE) return;

    const int col_tile  = blockIdx.x;   // 32-col tile index
    const int group_idx = blockIdx.y;   // row group index

    const int oc_base       = col_tile * TILE_COLS;
    const int ic_group_base = group_idx * group_size;

    if (ic_group_base >= IC) return;

    const int OC_packs = (OC + 7) / 8;
    const int num_groups = (IC + group_size - 1) / group_size;
    (void)num_groups;

    // Shared tile: [group_size, 32] FP16
    __shared__ half tile_w[MAX_GROUP_SIZE][TILE_COLS];
    // Per-tile scales for each column (oc) in this tile
    __shared__ half sm_scales[TILE_COLS];

    // -----------------------------------------------------------------
    // Phase 1: Load [group_size x 32] tile and compute max_abs per (group, oc)
    // -----------------------------------------------------------------
    float max_abs = 0.0f;

    for (int local_ic = 0; local_ic < group_size; ++local_ic) {
        const int ic = ic_group_base + local_ic;
        if (ic >= IC || local_ic >= MAX_GROUP_SIZE) break;

        const int oc = oc_base + lane;

        half w = __float2half(0.0f);
        if (lane < TILE_COLS && oc < OC) {
            const size_t idx = (size_t)ic * (size_t)OC + (size_t)oc;
            w = W[idx];
            const float fv = __half2float(w);
            const float a  = fabsf(fv);
            max_abs = fmaxf(max_abs, a);
        }

        if (lane < TILE_COLS) {
            tile_w[local_ic][lane] = w;
        }
    }

    const int oc_lane = oc_base + lane;
    float scale_f = (max_abs > 1e-8f) ? (max_abs / 7.0f) : 1.0f;

    if (lane < TILE_COLS) {
        sm_scales[lane] = __float2half(scale_f);

        if (oc_lane < OC) {
            const size_t sidx = (size_t)group_idx * (size_t)OC + (size_t)oc_lane;
            scales[sidx] = __float2half(scale_f);
        }
    }

    __syncthreads();

    // -----------------------------------------------------------------
    // qzeros: per (group, output_block) packed like qweight; v1 = all zeros
    // Only need to set once per (group_idx, oc_block). Do it here per tile.
    // oc blocks in this 32-col tile: pack_id = 0..3, oc0 = oc_base + 8*pack_id
    // -----------------------------------------------------------------
    if (ic_group_base < IC && lane < 4) {
        const int pack_id = lane;
        const int oc0 = oc_base + pack_id * 8;
        if (oc0 < OC) {
            const int pack_col = oc0 / 8;
            if (pack_col < OC_packs) {
                const size_t zidx = (size_t)group_idx * (size_t)OC_packs + (size_t)pack_col;
                qzeros[zidx] = 0;  // symmetric quantization: zero zero-points
            }
        }
    }

    __syncthreads();

    const unsigned full_mask = __ballot_sync(__activemask(), true);

    // Nibble order [0,2,4,6,1,3,5,7] for vLLM packing
    const int nibble_order[8] = {0, 2, 4, 6, 1, 3, 5, 7};

    // -----------------------------------------------------------------
    // Phase 2: Process group in 32-row sub-blocks.
    //
    // For each sub-block (32 rows), we:
    //   - load a 32x32 tile from shared into registers (per lane = column),
    //   - quantize into signed int4 (stored as int),
    //   - run warp-cooperative butterfly transpose,
    //   - then each lane = row, T[col] = q(ic, oc_offset),
    //   - pack 4 int32 blocks (8 outputs each) per lane (per ic).
    // -----------------------------------------------------------------
    for (int sub = 0; sub < group_size; sub += WARP_SIZE) {
        if (sub >= group_size) break;
        if (ic_group_base + sub >= IC) break;

        // 2.1 Load + quantize this 32x32 tile into registers: T[k] per lane
        int T[WARP_SIZE];

        #pragma unroll
        for (int k = 0; k < WARP_SIZE; ++k) {
            const int local_ic = sub + k;
            const int ic = ic_group_base + local_ic;

            half w_h = __float2half(0.0f);
            if (local_ic < group_size && local_ic < MAX_GROUP_SIZE &&
                lane < TILE_COLS && ic < IC) {
                const int oc = oc_base + lane;
                if (oc < OC) {
                    w_h = tile_w[local_ic][lane];
                }
            }

            const float w_f = __half2float(w_h);
            const float s_f = __half2float(sm_scales[lane]);

            // Symmetric signed INT4 quantization in [-8..7]
            float qf = (s_f > 0.0f) ? (w_f / s_f) : 0.0f;
            qf = nearbyintf(qf);
            qf = fminf(fmaxf(qf, -8.0f), 7.0f);

            const int qi = static_cast<int>(qf);
            T[k] = qi;  // T[k] = q(ic_sub = k, oc_lane = lane)
        }

        // 2.2 Warp-cooperative 32x32 butterfly transpose in registers.
        // After this:
        //   lane = row_in_sub (ic_sub)
        //   T[col] = q(ic_sub = lane, oc_offset = col)
        butterfly_transpose_32(T, full_mask);

        // 2.3 Each lane packs 4 int32 blocks for its ic
        const int ic = ic_group_base + sub + lane;
        if (ic >= IC) {
            // This lane corresponds to a row beyond IC; skip writes
            continue;
        }

        #pragma unroll
        for (int pack_id = 0; pack_id < 4; ++pack_id) {
            const int oc0 = oc_base + pack_id * 8;
            if (oc0 >= OC) continue;

            const int pack_col = oc0 / 8;
            if (pack_col >= OC_packs) continue;

            uint32_t packed = 0u;

            #pragma unroll
            for (int nib = 0; nib < 8; ++nib) {
                const int j_in_pack = nibble_order[nib];   // 0,2,4,6,1,3,5,7
                const int col_off   = (oc0 - oc_base) + j_in_pack; // 0..31
                if (col_off < 0 || col_off >= TILE_COLS) continue;

                const int qi = T[col_off];
                const uint32_t nibble = static_cast<uint8_t>(qi) & 0xF;
                packed |= (nibble << (4 * nib));
            }

            const size_t qidx =
                (size_t)ic * (size_t)OC_packs + (size_t)pack_col;
            qweight[qidx] = static_cast<int32_t>(packed);
        }

        __syncwarp(full_mask);
    }
}

// ---------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------
void forge_awq_v2_launch(
    torch::Tensor weights,   // [IC, OC], float16, CUDA
    torch::Tensor qweight,   // [IC, OC_packs], int32, CUDA
    torch::Tensor scales,    // [G,  OC], float16, CUDA
    torch::Tensor qzeros,    // [G,  OC_packs], int32, CUDA
    int group_size)          // e.g. 128
{
    TORCH_CHECK(weights.is_cuda(), "weights must be CUDA");
    TORCH_CHECK(qweight.is_cuda(), "qweight must be CUDA");
    TORCH_CHECK(scales.is_cuda(),  "scales must be CUDA");
    TORCH_CHECK(qzeros.is_cuda(),  "qzeros must be CUDA");

    TORCH_CHECK(weights.dim() == 2, "weights must be [IC, OC]");
    TORCH_CHECK(weights.scalar_type() == torch::kHalf,
                "weights must be float16");

    TORCH_CHECK(qweight.scalar_type() == torch::kInt32,
                "qweight must be int32");
    TORCH_CHECK(scales.scalar_type() == torch::kHalf,
                "scales must be float16");
    TORCH_CHECK(qzeros.scalar_type() == torch::kInt32,
                "qzeros must be int32");

    const int IC = static_cast<int>(weights.size(0));
    const int OC = static_cast<int>(weights.size(1));

    TORCH_CHECK(group_size > 0 && group_size <= MAX_GROUP_SIZE,
                "group_size must be in (0, ", MAX_GROUP_SIZE, "]");

    const int OC_packs = (OC + 7) / 8;
    const int G        = (IC + group_size - 1) / group_size;

    TORCH_CHECK(qweight.size(0) == IC,
                "qweight.size(0) must equal IC");
    TORCH_CHECK(qweight.size(1) == OC_packs,
                "qweight.size(1) must be ceil(OC/8)");

    TORCH_CHECK(scales.size(0) == G,
                "scales.size(0) must equal num_groups = ceil(IC/group_size)");
    TORCH_CHECK(scales.size(1) == OC,
                "scales.size(1) must equal OC");

    TORCH_CHECK(qzeros.size(0) == G,
                "qzeros.size(0) must equal num_groups = ceil(IC/group_size)");
    TORCH_CHECK(qzeros.size(1) == OC_packs,
                "qzeros.size(1) must be ceil(OC/8)");

    dim3 block(WARP_SIZE, 1, 1);
    dim3 grid(
        (OC + TILE_COLS - 1) / TILE_COLS,   // 32-column tiles
        G                                   // row groups
    );

    const half*  W_ptr       = reinterpret_cast<const half*>(weights.data_ptr<at::Half>());
    int32_t*     qweight_ptr = qweight.data_ptr<int32_t>();
    half*        scales_ptr  = reinterpret_cast<half*>(scales.data_ptr<at::Half>());
    int32_t*     qzeros_ptr  = qzeros.data_ptr<int32_t>();

    auto stream = at::cuda::getCurrentCUDAStream();

    forge_awq_v2_kernel<<<grid, block, 0, stream.stream()>>>(
        W_ptr,
        qweight_ptr,
        scales_ptr,
        qzeros_ptr,
        IC,
        OC,
        group_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in forge_awq_v2_kernel: %s\n",
               cudaGetErrorString(err));
    }
}

// ---------------------------------------------------------------------
// PyBind
// ---------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forge_awq_v2",
        &forge_awq_v2_launch,
        "4Bit Forge AWQ W4 v2 (warp-cooperative butterfly transpose/pack)"
    );
}
