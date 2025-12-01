# 4BIT FORGE: Implementation Specification & Kernel Reference

**Version:** 1.3  
**Context:** DeepSeek-Math-V2 pruning (671B → ~100B) on 1×H100  
**Role:** Low-level implementation blueprint for calibration, pruning, quantization and runtime integration with vLLM AWQ.

---

## 0. Global Design

### 0.1 Objective

Compress a 671B MoE checkpoint into a ~100B math-specialized variant that:

- Runs on a single H100 (80 GB).
- Uses vLLM’s AWQ W4A16 inference path.
- Requires **no** changes to vLLM’s C++ MoE kernels.

### 0.2 Core Constraints

- The full 1.3 TB checkpoint never resides in GPU memory.
- Pipeline is **streaming**:
  - Calibration: layer-wise, no gradients, sparse forward.
  - Forge: layer-wise, quantize and write out in AWQ format.
- Quantization:
  - 4-bit weights (W4).
  - Grouped scales, zero points (AWQ).
  - Output must match vLLM’s `qweight`, `scales`, `qzeros` layout exactly.

### 0.3 Main Components

1. **Calibration Streamer (`calibration/`)**  
   - Builds `hit_map[L, E]` with Sigmoid router statistics.  
   - Uses NVMe as activation swap if needed.

2. **Forge Core (CUDA kernels, `csrc/`)**  
   - Fused FP16 → INT4 + AWQ packing kernel.  
   - PackBoost-style warp, memory, and bit-packing tricks.

3. **Forge Orchestrator (`forge/`)**  
   - Builds `expert_map[L, 256]`.  
   - Streams layers, applies kernel, writes AWQ tensors.

4. **Runtime Adapter (`runtime/`)**  
   - Monkey-patches vLLM’s fused MoE path.  
   - Remaps original expert ids → compact ids, handles pruned experts.

---

## 1. Calibration Streamer (`calibration/`)

### 1.1 Goal

Estimate per-layer, per-expert importance for math without:

- Exceeding 80 GB VRAM.
- Running full MoE forwards.

We run in `torch.no_grad()`, skip expert MLP compute, and approximate next-layer routing from attention + residual paths.

### 1.2 `LayerwiseStreamer`

A controller for:

- Checkpoint metadata.
- Layer-wise streaming.
- Activation offloading.
- Expert hit statistics.

#### 1.2.1 Construction

Inputs:

- `model_path`: HF/safetensors root.
- `offload_dir`: directory on fast NVMe.
- `cfg`: calibration config (`num_layers`, `num_experts`, device, dtype, etc.).

Responsibilities:

- Read HF config to know:
  - Number of transformer layers.
  - Hidden size.
  - MoE/router structure.
- Read safetensors index to locate layer shards.
- Initialize:

  - `hit_map[L, E]` on CPU (float32, zeros).
  - `offload_dir` as activation swap directory:
    - For large calibration setups, activations per layer/batch can be written to
      `offload_dir/hs_layer_{l}_batch_{b}.pt` and reloaded.

#### 1.2.2 `stream_layers()`

Provides a “load → yield → free” protocol:

- For `layer_idx` in `[0 .. num_layers-1]`:
  - Resolve all tensors for `model.layers.{layer_idx}` (or equivalent path).
  - Load weights **only for that layer** (via `accelerate` or safetensors).
  - Move to `cuda:0`, using asynchronous copies where possible to overlap with I/O.
  - Yield `(layer_idx, layer_module)` to the caller.
  - Once the caller finishes:
    - Delete `layer_module`.
    - Call `torch.cuda.empty_cache()` to reclaim VRAM and reduce fragmentation.

Contract: the caller does not hold references to previous layers when the next iteration starts.

#### 1.2.3 Activation Handling

Two modes:

1. **Host RAM only**  
   - `hidden_states` stored in CPU tensors between layers.
   - For each layer:
     - Move current activations to GPU.
     - Run forward.
     - Move next activations back to CPU.

2. **NVMe offload**  
   - When host RAM is insufficient, full `hidden_states` or layer outputs are written to `offload_dir` as `.pt` or chunked files.
   - Only the needed slice for the current layer/batch is loaded into CPU/GPU at any time.

Implementation detail is flexible; the spec requires:

- No assumption that all calibration activations fit in GPU.
- A clear path to fall back to disk.

### 1.3 Router Stats Capture

#### 1.3.1 Router Discovery

Each transformer layer has a Sigmoid router. The streamer must be able to locate it:

- A configuration value like `cfg.router_attr = "mlp.moe.gate"` or similar.
- Resolution via chained `getattr(layer, "mlp")`, then `"moe"`, then `"gate"`.

If the architecture varies, you keep a small registry mapping layer classes → router attributes.

#### 1.3.2 Sigmoid-based probabilities

Given input `hidden_states` `[B, T, H]`:

- `logits = router(hidden_states)` → `[B, T, E]`.
- `probs = sigmoid(logits)`.

Key point:

- Softmax routers produce “winner-take-all-ish” distributions.  
- Sigmoid routers treat each expert as an independent Bernoulli gate:
  - Multiple experts can be strongly “on” at once.
  - Good for capturing nuanced roles and for math specialization signals.

The calibration logic must rely on these Sigmoid probabilities, not convert to a Softmax.

#### 1.3.3 Hit map accumulation

For each layer `l`:

- Compute per-expert mass:

  - `mass = probs.sum(dim=(0, 1))` → `[E]` (float32).

- Accumulate into CPU `hit_map`:

  - `hit_map[l] += mass.to("cpu")`.

This compresses `[B, T, E]` into `[E]` per layer, leaving only a small `[L, E]` tensor in RAM.

### 1.4 Sparse Forward Approximation

Goal: produce `next_hidden_states` for use by the next layer’s router without paying full MoE cost.

Assumption:

- In DeepSeek-style residual transformers, next-layer routing is dominated by:
  - Self-attention output.
  - Residual connection.
- Expert MLP outputs add detail but do not drastically change which experts fire.

Implementation pattern:

- For calibration only:

  - Run attention + residual + layernorm components.
  - Skip expert MLP computations entirely, or approximate them cheaply.
  - Use the resulting `next_hidden_states` as input to the next layer’s router.

This yields a high-fidelity `hit_map` with much less compute and no expert-activation VRAM blowup.

---

## 2. Forge Core: Quantization Kernel (`csrc/`)

### 2.1 Goal

A fused CUDA kernel that:

- Reads FP16 weights once.
- Computes groupwise max-abs / scale.
- Quantizes to signed INT4.
- Packs into vLLM’s interleaved `int32` layout.
- Writes `qweight` and per-group scales (and optionally zero-points) with minimal memory traffic.

Target: effective throughput close to H100 memory bandwidth (~copy-speed).

### 2.2 Logical Interface

Internal kernel view:

- Input matrix: `weights_fp16` `[rows, cols]`, row-major, FP16, where:
  - `rows = in_features` (IC).
  - `cols = out_features` (OC).
- Group size: `group_size` (e.g., 128) along the **input** (row) dimension for AWQ.

Outputs (logical):

- `int4` weights per element (signed, in range `[-8 .. 7]`).
- One scale per `(row_group, col)`.
- One zero-point per `(row_group, col)` or per group if needed.

These must be mapped to vLLM’s physical layouts described in section 3.

### 2.3 Threading and Tiling

Basic mapping:

- One warp processes one **quantization group** of `group_size` elements along rows for a given output column slice.
- Define:
  - `warps_per_row = cols / 8` groups of 8 columns, or a more general tile division.
  - `total_warps = rows * warps_per_row`.
- Warp assignment:
  - `warp_id` → `(row_index or row_group_index, col_group_index)` depending on exact tiling.
  - `lane_id ∈ [0..31]` inside each warp.

The kernel uses warp-aligned grid strides (block sizes multiples of 32) so all memory accesses are 128-byte aligned and easily coalesced. Tails (non-multiple sizes) are handled via lane-level masks, not irregular global memory ranges.

### 2.4 Vectorized Tile Loading

Per warp:

- Load a tile of FP16 weights into registers using vectorized global loads:
  - `float4` corresponds to 8 FP16 values when reinterpret-cast appropriately.
  - Consecutive lanes read consecutive `float4` words.

This:

- Maximizes memory bus utilization.
- Reduces instruction count compared to scalar loads.

No intermediate global writes occur before quantization.

### 2.5 Warp-Level Max-Abs Reduction

Per quantization group:

- Each lane computes `thread_max` as the maximum absolute value of the weights it owns.
- Use warp-shuffle XOR reduction:

  - Repeatedly combine `thread_max` across lanes with `max` and `__shfl_xor_sync`.

After the loop:

- Every lane holds `max_abs` for that group.
- Compute `scale = max_abs / 7.0` (avoid division by zero with a small epsilon).

One lane (e.g., lane 0) is responsible for writing this scale to the output scale tensor for the `(input_group, output_channel)` pair.

### 2.6 Quantization in Registers

For each FP16 weight:

- Convert to float.
- Compute:

  - `q = round(w / scale)` as float.
  - Clamp `q` to integer `[-8 .. 7]`.

- Convert to 4-bit:

  - `q4 = q & 0xF` (two’s complement mapping supplies correct nibble patterns if consistent with dequantizer).

These 4-bit values remain in registers until they are packed into `int32`.

If zero-points are used (AWQ does), they can be defined and stored similarly:

- AWQ often uses learned or derived zero-points per `(group, col)`; the kernel can either:
  - In v1: ignore zeros (center weights around zero and use symmetric quantization).
  - In AWQ-strict mode: compute or load zero-points and pack them into `qzeros` with the same pattern as `qweight`.

### 2.7 Interleaved Bit Packing

The packing must match vLLM’s AWQ format exactly:

- For each input channel (or group, depending on tiling), and for each block of 8 output channels `[c..c+7]`, we have 8 INT4 values:

  - `q0 = weight(ic, c)`
  - `q1 = weight(ic, c+1)`
  - …  
  - `q7 = weight(ic, c+7)`

These are inserted into a single `int32` using nibble order:

- Bits `0–3`   ← `q0`
- Bits `4–7`   ← `q2`
- Bits `8–11`  ← `q4`
- Bits `12–15` ← `q6`
- Bits `16–19` ← `q1`
- Bits `20–23` ← `q3`
- Bits `24–27` ← `q5`
- Bits `28–31` ← `q7`

Equivalently, the nibble index ordering is `[0, 2, 4, 6, 1, 3, 5, 7]`.

Packing is done in registers, using simple shifts and ORs.

### 2.8 PackBoost-Style Shared Memory Transpose (Optional)

If a direct register-only path is not convenient, an optional shared-memory staging step can be used:

1. For a warp handling a tile of quantized nibbles, write them to shared memory using **skewed indices**:
   - `smem[row][(col + row) & 31]`.

2. When reading out in the required interleaved column order (`0,2,4,6,1,3,5,7`), the skew ensures accesses are bank-conflict-free.

3. Pack in registers and store a single `int32`.

This is the same pattern used in PackBoost’s `encode_cuts.cu` to avoid shared-memory bank conflicts for 32×32 tiles.

### 2.9 Warp-Aligned Grid Strides and Templates

- All loops over linear indices (rows×col_groups) must stride in multiples of `warp_size * num_warps_per_block`.
- Tail elements are handled with lane-level conditionals but memory ranges are always aligned.
- The kernel can be templated on `PackedT` (e.g. `uint32_t`) if future formats demand different packing widths. v1 uses `uint32_t` consistently.

---

## 3. vLLM AWQ Layout Requirements

The Forge Core and Forge Orchestrator must already target the exact vLLM AWQ W4 layout.

Assume for each linear layer:

- Logical weight matrix: `W[OC, IC]` in PyTorch.
- vLLM’s AWQ format expects things transposed into `[IC, OC]` at the packing stage.

### 3.1 `qweight`

- **dtype:** `int32`
- **shape:** `[IC, OC // 8]`
- Each entry packs 8 W4 values for a fixed input channel across 8 consecutive output channels.

Nibble mapping per int32:

- Interpret `int4_tensor[ic, oc]` as un-packed values in `[0..15]`.
- For output channels `c..c+7`:

  - `q0 = int4(ic, c)`
  - `q1 = int4(ic, c+1)`
  - …
  - `q7 = int4(ic, c+7)`

Packed as:

- Bits 0–3   → `q0`
- Bits 4–7   → `q2`
- Bits 8–11  → `q4`
- Bits 12–15 → `q6`
- Bits 16–19 → `q1`
- Bits 20–23 → `q3`
- Bits 24–27 → `q5`
- Bits 28–31 → `q7`

The Forge kernel is responsible for producing this **directly**, without an extra permute kernel.

### 3.2 `scales`

- **dtype:** `float16`
- **shape:** `[IC // group_size, OC]`
- Interpretation:

  - Group index `g = floor(ic / group_size)`.
  - `scales[g, c]` is the scale used for all inputs `ic ∈ [g*group_size .. g*group_size + group_size - 1]` into output channel `c`.

Forge mapping:

- If the kernel writes per-row per-group scales as `[IC, OC/group_size]` for convenience, the Forge Orchestrator must reshape and aggregate them into `[IC // group_size, OC]` consistent with vLLM’s expectation.

### 3.3 `qzeros`

- **dtype:** `int32`
- **shape:** `[IC // group_size, OC // 8]`
- Packing is identical to `qweight`:

  - For each `(group, output_block)` pair, the 8 zero-point nibbles `[z0..z7]` are packed in the same `[0,2,4,6,1,3,5,7]` nibble layout.

Dequantization in vLLM uses:

- `fp16 = (qweight_int4 - qzero_int4) * scale_fp16`.

For 4BIT FORGE v1:

- You can implement symmetric quantization (zero-point = mid-point) or full AWQ zero-points, but the layout must still match above shapes and packing.

---

## 4. Forge Orchestrator (`forge/`)

### 4.1 Goals

- Turn `hit_map[L, 256]` into a per-layer pruning decision.
- Build a compact `expert_map[L, 256]`.
- Stream the original checkpoint and produce:
  - Dense layers in AWQ (`qweight`, `scales`, `qzeros`).
  - Per-layer compact expert weights in AWQ format.
- Persist all artifacts as SafeTensors with vLLM-compatible naming and shapes.

### 4.2 Expert Map Construction

Given:

- `hit_map[L, 256]` with accumulated Sigmoid mass per expert.

For each layer `l`:

1. Sort experts by `hit_map[l, :]` descending.
2. Select top `keep_k` indices (`keep_indices`).
3. Create `layer_map[256]` initialized with `-1`.
4. Assign:

   - For each rank `r` in `0..keep_k-1`:
     - `orig_eid = keep_indices[r]`.
     - `layer_map[orig_eid] = r`.

Stack per-layer maps into `expert_map[L, 256]`.

Persist `expert_map.pt` for runtime use.

### 4.3 Layer Forge

For each transformer layer `l`:

1. Load FP16 weights for layer `l` from original checkpoint.
2. Identify:
   - Dense linear weights (attention, projections, shared MLP).
   - MoE expert weights (per expert id).

#### 4.3.1 Dense linear weights

For each dense linear weight `W_dense`:

- Logical shape `[out_features, in_features]`.
- Transpose to `[in_features, out_features]` if needed for the kernel.
- Call the quantization kernel to produce:

  - `qweight[IC, OC//8]`
  - `scales_raw` and optional `qzeros_raw` in some kernel-friendly internal shape.

- Reshape or reorder internal outputs into vLLM AWQ `qweight`, `scales`, `qzeros` layouts:
  - `qweight[IC, OC//8]` as already packed.
  - `scales[IC//G, OC]` derived from raw scale layout.
  - `qzeros[IC//G, OC//8]` if zero-points are supported.

- Store these three tensors with the names expected by vLLM’s AWQ loader for that layer (e.g. attention projections, shared experts).

#### 4.3.2 Expert weights

For each expert layer `l`:

- For each original expert `orig_eid in [0..255]`:
  - `compact_id = expert_map[l, orig_eid]`.
  - If `compact_id == -1`: skip (pruned).
  - If `compact_id >= 0`:
    - Load FP16 weights for expert `orig_eid` (gate/up/down).
    - Transpose to `[in_features, out_features]` if needed.
    - Quantize using kernel.
    - Place resulting `qweight`, `scales`, and `qzeros` into compact structures indexed by `compact_id`.

At the end for that layer:

- You have compact expert tensors, e.g.:

  - `qweight_experts[keep_k, IC, OC//8]` or some other packed shape depending on how vLLM expects MoE expert storage.
  - `scales_experts[keep_k, IC//G, OC]`.
  - `qzeros_experts[keep_k, IC//G, OC//8]`.

These must be reshaped/sliced and named consistent with vLLM’s MoE loader (e.g. `experts.{id}.up_proj.qweight`, etc.).

---

## 5. Runtime Adapter (`runtime/`)

### 5.1 Goal

Make vLLM run the pruned, compact expert model without modifying its C++ fused MoE kernels.

Strategy:

- Use standard vLLM AWQ loading for the forged checkpoint.
- Monkey-patch the fused MoE call to:

  - Compute routing as usual (Sigmoid + top-k).
  - Remap original expert ids to compact ids via `expert_map`.
  - Redirect pruned experts to a safe expert (0).
  - Zero out their routing weights.

### 5.2 Expert Map Usage

At startup:

- Load `expert_map.pt` into GPU as `[L, 256]`.
- Establish a mapping `layer_index_in_vllm → l` to index the correct slice.

Guarantee:

- For every layer, at least one expert is kept (`keep_k > 0`), so compact id 0 is valid.

### 5.3 Patched MoE Forward

For a given MoE layer:

1. Compute routing:

   - `routing_weights = sigmoid(router_logits)`  
   - `topk_weights, topk_ids = topk(routing_weights, top_k, dim=-1)`

   Here, `topk_ids` are in `[0..255]` (original expert ids).

2. Flatten ids and weights:

   - `flat_ids = topk_ids.view(-1)`
   - `flat_weights = topk_weights.view(-1)`

3. Remap:

   - `layer_map = expert_map[layer_idx]` `[256]` on GPU.
   - `remapped = layer_map[flat_ids]` → `[B*T*top_k]`

4. Handle pruned ids:

   - `mask = (remapped == -1)`
   - `remapped[mask] = 0`  (route to expert 0)
   - `flat_weights[mask] = 0.0`  (so redirected traffic contributes nothing)

5. Reshape back:

   - `remapped_ids = remapped.view_as(topk_ids)`
   - `topk_weights_reshaped = flat_weights.view_as(topk_weights)`

6. Call the original fused MoE CUDA kernel with:

   - `topk_ids = remapped_ids`
   - `topk_weights = topk_weights_reshaped`
   - `renormalize = True` (to ensure remaining non-zero weights sum to 1 per token).

The C++ kernel remains unaware of pruning; it sees a dense compact expert space per layer.

---

## 6. Future Work: W4A16 GEMM (Optional v2)

4BIT FORGE v1 reuses vLLM’s AWQ GEMM kernels. A natural extension:

- Implement a custom W4A16 GEMM kernel that:

  - Reads `qweight` and `qzeros` in the AWQ layout.
  - Unpacks 8 INT4 values per int32 in registers using the **same** nibble mapping (`0,2,4,6,1,3,5,7`).
  - Immediately applies `(w_int4 - z_int4) * scale_fp16`.
  - Feeds dequantized FP16 blocks into Tensor Cores via `wmma` for matmul.
  - Uses warp-shuffle reductions along K to avoid global atomic contention.

PackBoost patterns apply directly:

- Warp-aligned grid strides.
- Shared memory tiling with skewed indexing to avoid bank conflicts.
- Register-level accumulation and reduction.

The current spec keeps this out of v1, but it informs how you structure tiling and data layouts so they are future-compatible.

---

## 7. Validation and Invariants

The implementation must satisfy:

1. `hit_map[L, E]` is computed using Sigmoid router probabilities with expert MLPs skipped and attention+residual approximation.
2. `expert_map[L, 256]` is dense on `[0..keep_k-1]` and uses `-1` for pruned experts.
3. Quantization kernel is **bit-exact** against a Python reference implementing:
   - Per-group max-abs scale.
   - Signed INT4 range.
   - AWQ nibble ordering `[0,2,4,6,1,3,5,7]`.
4. Output tensors for each linear weight match vLLM’s AWQ specs:
   - `qweight[IC, OC//8]` (int32).
   - `scales[IC//G, OC]` (fp16).
   - `qzeros[IC//G, OC//8]` (int32).
5. Patched MoE routing:
   - Never passes a negative or OOB expert index to C++ kernels.
   - Maps pruned experts to expert 0 with zero weight.
   - Produces numerically stable outputs for non-pruned expert routes.

This document is the reference blueprint for anyone touching the calibration streamer, CUDA kernels, forge orchestration, or vLLM runtime patching in 4BIT FORGE.
