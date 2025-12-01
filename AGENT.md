# 4BIT FORGE – Internal Reference Spec

DeepSeek-Math-V2 Expert Pruning + 4-bit AWQ Quantization Pipeline
Target: ~100B math-specialized dense-ish model runnable on 1×H100 using vLLM.

This document gives a **mental model of the repo** in terms of components, data structures, and control flows, without going into implementation details.

---

## 1. Top-Level Conceptual Layout

4BIT FORGE is composed of six conceptual subsystems:

1. **Checkpoint I/O Layer**
   Reads and writes DeepSeek-Math-V2 weights as a stream of layer chunks. Provides a uniform interface over SafeTensors / shard formats.

2. **Calibration Streamer**
   Runs a no-grad, layer-wise forward pass over math calibration data. Produces an `Expert_Hit_Map` for all MoE layers.

3. **Expert Stats & Pruning Logic**
   Interprets `Expert_Hit_Map` into per-layer `Keep_List`s and a global `Expert_Translation_Table`.

4. **Quantization & Packing Kernel Layer (“Forge Core”)**
   Exposes a fused INT4 AWQ kernel driven by PackBoost-style warp tiling + bit packing. This is the performance-critical engine.

5. **Forge Orchestrator**
   Drives checkpoint streaming, pruning decisions, and quantization kernel calls to produce a new pruned+quantized checkpoint plus the expert map.

6. **Runtime Adapter for vLLM**
   A Python-only shim that remaps original expert ids to compact ids before calling vLLM’s fused MoE C++ kernels.

Think of it as:

* **Streamer**: “Which experts matter?”
* **Pruner + Forge Core**: “Keep only those, compress them into INT4 AWQ format.”
* **Adapter**: “Lie to vLLM so it thinks the pruned thing is a native dense model.”

---

## 2. Canonical Data Structures

These are the core logical data structures that everything revolves around.

### 2.1 Model Geometry

* `num_layers` (int): total transformer blocks (≈ 61 for DeepSeek-Math-V2).
* `num_moe_layers` (int): subset of layers that are MoE.
* `num_experts` (int): experts per MoE layer (e.g., 256).
* `top_k` (int): router top-k (e.g., 4 or 8).
* `group_size` (int): AWQ quantization group size along columns (e.g., 128 or 256).

These are read from the original model config and reused in calibration, pruning, and runtime.

### 2.2 Expert Hit Map

* `Expert_Hit_Map`: tensor with shape `[num_moe_layers, num_experts]`, dtype float32.
  For MoE layer index `ℓ` and expert index `e`:

  * `Expert_Hit_Map[ℓ, e]` stores accumulated gating mass or selection score over calibration data.

Semantics:

* Higher value → expert receives more math routing.
* This is “ground truth” for pruning decisions.

### 2.3 Per-Layer Expert Selection

For each MoE layer `ℓ`:

* `Keep_List_ℓ`: ordered list of expert indices to keep.
* `K_keep_ℓ` = length of `Keep_List_ℓ` (e.g., 32).

Optional helpers:

* `Keep_Mask_ℓ`: boolean array of length `num_experts`, true if expert kept.

### 2.4 Expert Translation Table

Single source of truth for mapping original expert ids → compact ids.

Two equivalent representations:

* **Per-layer 2D tensor**:
  `Expert_Map[ℓ, e]` ∈ `{0 .. (K_keep_ℓ - 1)} ∪ {-1}`

  * If expert `e` in layer `ℓ` is kept, holds its compact index.
  * If pruned, value is `-1`.

* **Flattened 1D tensor for runtime**:
  `EXPERT_MAP_TENSOR` of length `num_moe_layers * num_experts`, with a known offset rule like:
  `flat_index = ℓ * num_experts + e`.

This table is:

* Produced by the pruning subsystem.
* Serialized to disk as `expert_map` (e.g., `.pt`).
* Loaded by the runtime adapter and moved to CUDA.

### 2.5 Quantized Weights Layout (AWQ)

For any quantized weight matrix:

* Original FP16 shape: `[rows, cols]`.
* Grouped along `cols` by `group_size`:

  * `num_groups = cols / group_size` (must divide exactly).

Outputs:

1. **Packed INT4 values**

   * Tensor shape: `[rows, cols / 8]` (8 4-bit values per 32-bit word).
   * Layout: AWQ column-interleaved arrangement compatible with vLLM’s dequantization kernel.

2. **Group scales**

   * Tensor shape: `[rows, num_groups]`.
   * dtype: FP16 or FP32.
   * `scale(r, g)` = max_abs of group `(r, g)` divided by 7.

The precise nibble ordering inside each 32-bit word is defined by the AWQ layout investigation and must match vLLM.

---

## 3. Checkpoint I/O Layer

This layer abstracts away “1.3TB checkpoint” into manageable layer-level chunks.

### 3.1 Responsibilities

* Map model configs (layer types, parameter names) to on-disk shard names.
* Efficiently load a **single transformer layer** (attention + MLP + MoE) into host or device memory.
* Provide streaming iteration over:

  * All layers in order.
  * Within a layer, all MoE experts (weights for each expert).
* Write out compact checkpoint:

  * Only kept experts.
  * Quantized + packed weights.
  * AWQ scales.

### 3.2 Key Operations

Conceptual operations (not function signatures):

* “Open original checkpoint for streamed reads.”
* “Load layer `ℓ` parameters as tensors.”
* “Load expert `e` parameters within layer `ℓ`.”
* “Write quantized weights/scales for layer `ℓ` into new checkpoint under compact indexing.”

Invariants:

* No operation attempts to bring the **full model** into GPU memory.
* Storage layout for the new checkpoint must be deterministic, derived from `Keep_List_ℓ`.

---

## 4. Calibration Streamer

This subsystem “walks” the model once, focusing on router behavior under math prompts.

### 4.1 Goals

* Run a forward pass that:

  * Evaluates MoE routers under math calibration workloads.
  * Records how much each expert is used.
* Avoid OOM:

  * Use `torch.no_grad`.
  * Only hold **one layer’s weights** plus current activations on GPU.

### 4.2 Inputs

* Model path and geometry.
* Calibration dataloader:

  * Math-heavy prompts.
  * Batches small enough to fit in memory but large enough to be representative.

### 4.3 Core Logic

For each MoE layer index `ℓ`:

1. **Current Activations**

   * Start with embeddings for the entire calibration batch.
   * For subsequent layers, re-use previous layer’s output activations.

2. **Layer Load**

   * Load layer `ℓ` weights to GPU (attention, router, experts).

3. **Router Evaluation**

   * Compute router logits `router_logits` from current activations.
   * Apply Sigmoid → `scores`.
   * Aggregate:

     * sum across batch and tokens → `expert_mass_ℓ` vector length `num_experts`.
     * Update `Expert_Hit_Map[ℓ, :] += expert_mass_ℓ`.

4. **Sparse Forward**

   * Advance activations while minimizing MoE compute:

     * Option A: exact MoE evaluation for top-k experts per token.
     * Option B: approximate by using residual + attention outputs, possibly omitting full MLP expert evaluation.
   * Output becomes input for layer `ℓ+1`.

5. **Cleanup**

   * Free layer weights from GPU.
   * Move to next layer.

No gradients or layers beyond the current one are live at any point.

### 4.4 Output Guarantees

* `Expert_Hit_Map` is fully populated for all MoE layers and experts.
* The streamer is deterministic given fixed calibration data and seeds.

---

## 5. Expert Stats & Pruning Subsystem

This subsystem translates `Expert_Hit_Map` into concrete pruning decisions and the translation table.

### 5.1 Responsibilities

* Interpret raw scores into per-layer keep lists.
* Generate the global `Expert_Translation_Table`.
* Optionally log statistics for inspection.

### 5.2 Selection Strategy

Default strategy (configurable):

* For each MoE layer `ℓ`:

  * Normalize scores if desired (e.g., divide by sum over experts).
  * Sort experts by descending score.
  * Take top `K_keep_ℓ` experts (a configurable constant or per-layer schedule).
  * `Keep_List_ℓ` is this ordered list.

### 5.3 Translation Table Construction

For each MoE layer `ℓ`:

* Initialize all entries as `-1` (pruned).
* For each kept expert index `e` in `Keep_List_ℓ` with position `k`:

  * Set `Expert_Map[ℓ, e] = k`.

Properties:

* `Expert_Map[ℓ, e]` is a dense 0..`K_keep_ℓ`–1 mapping for kept experts.
* Pruned experts stay at `-1` and never correspond to real weights in the compact checkpoint.

This table is serialized for reuse by:

* Forge Orchestrator: to know which experts to load and where to place them in the new checkpoint.
* Runtime Adapter: to remap router indices during inference.

---

## 6. Quantization & Packing Kernel Layer (Forge Core)

This is the C++/CUDA extension that performs high-throughput INT4 AWQ compression.

### 6.1 Scope

* Operates on generic weight matrices `[rows, cols]` in FP16.
* Produces:

  * Packed INT4 weights `[rows, cols / 8]`.
  * AWQ scales `[rows, cols / group_size]`.
* Layout and scale definition must be **bit-exact** with vLLM AWQ expectations.

### 6.2 Kernel Inputs/Outputs (Logical)

Inputs:

* `input_weights`: FP16 matrix `[rows, cols]`.
* `rows`, `cols` (ints).
* `group_size` (int, divides `cols`).
* Possibly additional stride/leading-dim info to match real layout.

Outputs:

* `output_packed`: INT32 matrix `[rows, cols / 8]` containing packed int4s.
* `output_scales`: FP16/FP32 `[rows, cols / group_size]` storing group-wise scales.

### 6.3 Quantization Logic

Per group (row `r`, group index `g`):

1. Collect `group_size` weights.
2. Compute `max_abs = max(|w|)`.
3. Compute scale `s = max_abs / 7.0`.
4. For each `w` in the group:

   * `q = round(w / s)`, clamped to [−8, 7].
5. Arrange `q` values into 4-bit unsigned representation (with offset as needed by AWQ) and pack into 32-bit words.

### 6.4 Warp-Level Tiling

* Each **warp** is assigned to a specific group (or fixed number of groups).
* Each thread in a warp:

  * Loads multiple FP16 values into registers.
  * Participates in max reduction via warp shuffles.
* After scale broadcast:

  * Each thread quantizes its local subset.
  * Threads exchange specific nibbles via shuffles so that each thread responsible for writing a 32-bit word has exactly the 8 nibbles it needs.

### 6.5 PackBoost Techniques Used

* **Warp reductions for max**: same pattern as PackBoost’s cut kernel.
* **Register-only bit assembly**:

  * Similar to repack logic:

    * Use shifts + ORs in registers, no round trips through shared/ global memory.
* **Memory behavior**:

  * Coalesced global loads of FP16.
  * Coalesced stores for packed INT32 and scales.

Performance constraints:

* Single pass over input (no re-read per group).
* No dynamic memory allocation per call.
* Results must be deterministic.

---

## 7. Forge Orchestrator

High-level driver that takes:

* Original DeepSeek-Math-V2 checkpoint.
* Expert translation table.
* AWQ quantization kernel.

And produces:

* New pruned+quantized checkpoint.
* Expert map artifact for runtime.

### 7.1 Responsibilities

* Iterate over all transformer layers in the original checkpoint.
* For each layer:

  * Quantize dense / shared (non-MoE) weights.
  * For MoE layers:

    * Consult `Keep_List_ℓ`.
    * Quantize only kept experts.
    * Allocate them contiguously in the compact checkpoint under compact indices.
* Ensure that all layout and metadata are consistent with the runtime adapter and vLLM.

### 7.2 Logical Processing per Layer

1. Load layer `ℓ` weights via I/O layer.
2. Non-MoE parts:

   * Feed into Forge Core, write packed weights/scales to output checkpoint.
3. MoE experts:

   * For each expert `e`:

     * If `Expert_Map[ℓ, e] == -1`: skip (pruned).
     * Else:

       * Quantize via Forge Core.
       * Place at expert index `Expert_Map[ℓ, e]` in output checkpoint.
4. Record any auxiliary metadata needed for vLLM:

   * For example, updated tensor shapes for expert dimensions.

The orchestrator **never** sees full-model activation data; it operates purely on weights and the expert map.

---

## 8. Runtime Adapter for vLLM

This is the “Adapter” that makes the compressed model executable by vLLM without C++ changes.

### 8.1 Purpose

* Intercept the MoE forward path.
* Remap expert ids from original indexing to compact indexing.
* Mask out pruned experts cleanly.

### 8.2 Inputs

At inference time per MoE call:

* `hidden_states`: token representations.
* `w1`, `w2`: compact expert weight tensors with shape driven by `max(K_keep_ℓ)`.
* `router_logits`: raw logits from router.
* `top_k`: router top-k.
* `EXPERT_MAP_TENSOR`: global mapping on GPU.

### 8.3 Logical Steps

Per MoE invocation:

1. **Compute routing weights**

   * Apply Sigmoid to router logits.

2. **Select original top-k**

   * Use vLLM’s existing top-k logic to get:

     * `topk_ids` (original expert ids).
     * `topk_weights`.

3. **Remap IDs**

   * Flatten `topk_ids`.
   * Index into `EXPERT_MAP_TENSOR` to get `remapped_ids`.
   * For entries where `remapped_ids == -1`:

     * Mark those as pruned.

4. **Handle pruned routes**

   * Set corresponding entries in `topk_weights` to 0.
   * Redirect `remapped_ids` entries from `-1` to a valid fallback expert (e.g., 0) to avoid C++ kernel indexing issues. Their weights are zero so they contribute nothing.

5. **Renormalization (optional)**

   * If necessary, renormalize `topk_weights` per token to sum to 1 over real experts.

6. **Call original fused MoE kernel**

   * Pass `hidden_states`, compact expert weights, remapped ids, and modified weights into vLLM’s fused_moe.

From vLLM’s perspective, it is working with a smaller dense set of experts; the existence of pruned experts is entirely hidden behind the remapping.

---

## 9. End-to-End Control Flows

### 9.1 Calibration Flow

1. Load model config + checkpoint metadata.
2. Initialize `Expert_Hit_Map` zeros.
3. For each calibration batch:

   * Embed text → initial activations.
   * For each layer:

     * Stream layer weights to GPU.
     * Run router.
     * Update `Expert_Hit_Map`.
     * Advance activations using sparse forward.
     * Free layer weights.
4. Save `Expert_Hit_Map` and calibration logs.

### 9.2 Forge Flow

1. Load `Expert_Hit_Map`.
2. Run pruning logic → `Keep_List_ℓ` for each layer.
3. Build `Expert_Translation_Table` and save `EXPERT_MAP_TENSOR`.
4. For each layer:

   * Stream layer weights.
   * Quantize dense parts with Forge Core → write to new checkpoint.
   * For each expert in this layer:

     * If mapped index >= 0:

       * Quantize + pack via Forge Core.
       * Write to appropriate compact expert slot.
   * Free layer weights.
5. Finalize new checkpoint.

### 9.3 Inference Flow

1. Load vLLM with new checkpoint.
2. Load `EXPERT_MAP_TENSOR` into GPU.
3. Apply runtime adapter patch to fused_moe.
4. Run generation / benchmarking as usual:

   * Router produces logits in original expert indexing convention.
   * Adapter remaps them on-the-fly to compact expert indices.
   * Fused MoE uses compact weights.

---

## 10. Invariants and Sanity Conditions

* `Expert_Hit_Map` must be finite and non-negative.
* For each layer, `Keep_List_ℓ` length must be > 0 and ≤ `num_experts`.
* For any `(ℓ, e)`:

  * If `Expert_Map[ℓ, e] >= 0`, then the compact checkpoint must contain weights for that `(ℓ, compact_index)`.
* Quantized weights and scales must round-trip through vLLM’s AWQ dequantization path without bit-layout mismatch.
* Runtime adapter must never produce `topk_ids` that index out of bounds of compact expert tensors.

