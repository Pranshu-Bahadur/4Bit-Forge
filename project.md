# 4BIT FORGE – Internal Reference Spec (v1: GPTQ Only)

4BIT FORGE v1 is a **GPTQ W4A16-G128 quantization engine** for large language models.

- Scope: **Quantize an existing FP8 LLM** to 4-bit GPTQ weights.
- I/O and runtime are **delegated to existing libraries**:
  - `transformers`, `torch`
  - `llmcompressor` (for model graph / pipelines)
  - `compressed-tensors` (for storage formats)
  - `vllm` (for runtime execution)
- 4BIT FORGE focuses on:
  - **Calibration stats collection (Hessian-like)**
  - **GPTQ solve**
  - **High-performance CUDA kernels for quantization + packing**

---

## 1. Top-Level Subsystems

4BIT FORGE v1 is organized into six conceptual subsystems:

1. **Frontend & Orchestrator (Python)**
2. **Checkpoint & Runtime I/O (Library Integrations)**
3. **Calibration Runner**
4. **Statistics Engine (Hessian / GPTQ Stats)**
5. **GPTQ Core (Blockwise Quantization Solver)**
6. **Quantization & Packing Kernels (“Forge Core”)**

Later versions can add AWQ, MoE pruning, multi-GPU, etc. v1 is “single-model GPTQ done right.”

---

## 2. Canonical Data Structures

### 2.1 Model Geometry

Global model descriptors (largely from `transformers` config):

- `num_layers` (int): number of transformer blocks.
- `hidden_size` (int): model hidden width.
- `num_heads`, `num_kv_heads` (ints): used to decide which matrices to quantize.
- `act_dtype` (e.g. `torch.float16`): activation precision during calibration.
- `target_dtype` for weights: 4-bit GPTQ (`W4A16`).

Per-layer geometry:

- `LayerDesc`:
  - `layer_idx` (int)
  - `name` (e.g. `"model.layers.12"` or HF path)
  - `weight_specs`: list of `WeightDesc` (see below)

### 2.2 Weight Descriptors

Not all parameters are treated equally. We describe each quantizable tensor with:

- `WeightDesc`:
  - `name`: full parameter name (e.g. `"mlp.down_proj.weight"`)
  - `shape`: `(out_features, in_features)`
  - `quantizable`: bool
  - `gptq_block_size` (usually `128`)
  - `symmetric` (bool; default `True` for v1)
  - `target_layout`: enum:
    - `GEMM_LINEAR` (generic matmul)
    - `MARLIN_4BIT`
    - `VLLM_GPTQ` (if we match a specific backend layout)

`WeightDesc` is produced by the I/O layer and consumed by GPTQ Core + Forge Core.

### 2.3 GPTQ Block Geometry

For each weight matrix `W` with shape `[OC, IC]` (out, in):

- We quantize along **input channels** in fixed-size groups:

  - `group_size = 128` (v1 default).
  - For each row `oc`, we split:
    - `IC` → `num_groups = ceil(IC / group_size)` blocks.

Per block:

- Block index: `(oc, g)` where `g ∈ [0, num_groups)`.
- Block weight view:
  - `w_block` has shape `[group_size]` (slice of that row).
- GPTQ statistics:
  - `H_block` $\in \mathbb{R}^{G×G}$ (G=group_size) or low-rank approx of it.
  - Optional `g_block` (first-order term) if needed.

Logical types:

- `H_block`: fp16/sec-order stats, stored as 128×128 per block.
- `g_block`: fp32 or fp16 vector length 128 (optional, depending on solver variant).

### 2.4 Calibration Stats Store

Per layer `L` and weight `W`:

- `Stats[L][W]`:
  - `H_blocks`: list of Hessian-like blocks, one per group.
    - Each `H_block` shape: `[group_size, group_size]`.
    - Stored fp16 or fp32 depending on config.
  - Optional:
    - `g_blocks`: list of 1st-order vectors `[group_size]`.

We assume a **blockwise GPTQ** design. This keeps memory manageable and allows:

- Single-pass accumulation over calibration data.
- All layer Hessians resident on GPU/CPU if desired.

### 2.5 Quantized Weight Representation

For each quantized tensor, we maintain:

- `qweight`: packed INT4 weights:
  - Shape depends on **target layout**:
    - Generic linear GEMM: `[OC, IC_packed]` where `IC_packed = ceil(IC / 8)` as `int32`.
    - Marlin / vLLM GPTQ layouts: possibly `[OC, IC_packed, ...]` according to backend.
- `scales`: FP16 or FP32 scale parameters:
  - Shape: `[OC, num_groups]` for blockwise GPTQ (one scale per block per row).
- Optional:
  - `zeros`: INT4/INT8 per block per row, same shape as `scales` or `[OC, num_groups]` but stored packed if backend expects it.

The exact packing (bit order, strides) is standardized for v1 and tied to a specific backend (e.g., Marlin).

---

## 3. Checkpoint & Runtime I/O (Library Integrations)

4BIT FORGE v1 **does not reinvent checkpoint reading or inference**. It wraps:

- **Loading original model**:
  - `transformers` + `torch.load`/SafeTensors.
  - Or `llmcompressor` abstractions if they expose model graph + weights.
- **Writing quantized weights**:
  - `compressed-tensors` (for producing a compact format).
  - Optional: vLLM-friendly weight packs.

### 3.1 Responsibilities

- Load a pretrained model (HF / custom) into host/GPU memory.
- Provide a **layer-wise iterable view**:

  - For each `LayerDesc`:
    - Expose its `WeightDesc`s and underlying tensors as `torch.Tensor`.
- Save quantized results:

  - Store `qweight`, `scales`, `zeros` for each `WeightDesc`.
  - Record meta info (block size, layout kind, dtype).

### 3.2 Dependencies

- `transformers` model classes (e.g. `AutoModelForCausalLM`).
- `torch` tensor API.
- `llmcompressor` for:
  - Pre-defined pipelines (optional).
  - Graph exploration, layer tagging, or hooking.

The I/O subsystem is “thin”: it identifies **what** to quantize and yields raw tensors; GPTQ and kernels decide **how**.

---

## 4. Calibration Runner

This subsystem executes the model in no-grad mode on a calibration dataset to collect **activation statistics** that feed GPTQ.

### 4.1 Inputs

- Loaded model (from I/O).
- Dataloader or generator of calibration prompts:
  - Might be built using `llmcompressor`’s calib pipeline or pure HuggingFace datasets.
- List of `WeightDesc` we care about.

### 4.2 Hook Strategy

For each linear weight `W` we will quantize, we need the **input activation vectors** hitting that matrix:

- For a linear `y = xWᵀ` with `W` shape `[OC, IC]`, we want the rows of `x` in shape `[batch * seq, IC]`.

Calibration runner:

- Register **forward hooks** at the right points in the model to capture `x` before the linear.
- To keep memory bounded:
  - We process in **micro-batches**.
  - We do **streaming accumulation** into Hessian blocks (see next section) instead of storing all `x`.

### 4.3 Execution Flow

For each calibration batch:

1. Tokenize prompts → input ids.
2. Run model forward with `torch.no_grad()`.
3. For each hooked linear:
   - Retrieve `x` (`[tokens, IC]`).
   - Immediately pass `x` to the **Statistics Engine** to update its per-block aggregates.
4. Discard `x` after stats update.

No gradients are computed; we only read activations.

---

## 5. Statistics Engine (Hessian / GPTQ Stats)

This is the numerical heart of calibration.

### 5.1 Goal

For each weight block of size `G = group_size`:

- Approximate the blockwise Hessian/Loss curvature:

  \[
    H_g \approx \sum_{i} x_{i,g} x_{i,g}^T
  \]

  where `x_{i,g}` is the slice of input vector for that group of input channels.

Optionally:

- Maintain first-order stats `g_g` for certain GPTQ variants:

  \[
    g_g \approx \sum_i x_{i,g} \cdot \epsilon_i
  \]

  (where `ε_i` is residual error term; v1 can initially ignore this or treat it with a simplified form.)

### 5.2 Streaming Update

For each minibatch and each hooked linear:

- Given `X` shape `[N, IC]`:

  1. Partition columns into groups of size `G`.
  2. For each group `g`:
     - Extract `X_g` shape `[N, G]`.
     - Compute `H_g += X_gᵀ @ X_g`.

To avoid naive Python loops, 4BIT FORGE will introduce a **custom CUDA kernel** or a batched GEMM-based implementation:

- Option A (baseline): use `torch.matmul` in a loop over groups.
- Option B (Forge-specific): a kernel that:
  - Tiles `X` and accumulates `X_gᵀX_g` for multiple groups in parallel.
  - Uses shared memory and warp-level ops (PackBoost-ish style).

### 5.3 Storage & Precision

Per block `H_g`:

- Stored as 128×128 matrix.
- Precision policy:
  - Accumulate in fp32, optionally downcast to fp16 after calibration.
- Aggregation can be kept on GPU; final solver may pull to CPU or operate in-place on GPU.

---

## 6. GPTQ Core (Blockwise Quantization Solver)

This subsystem solves the GPTQ optimization per block using `H_g` (and optionally `g_g`) and the original weights `W`.

### 6.1 Objective (Conceptual)

For each row `w` and each group `g`:

- You want an INT4 vector `q_g` such that:

  \[
    $\min_{q_g} \; (w_g - D(q_g))^T H_g (w_g - D(q_g))$
  \]

  where `D` is dequantize given scale+zero.

Standard GPTQ heuristics:

- Order channels by importance (e.g., diagonal of `H_g`).
- Greedy update with error compensation.

v1 doesn’t need to re-derive the paper; it just needs:

- A numerically stable implementation of **blockwise GPTQ** on top of `H_g` and `w_g`.
- Config switch:
  - `solver_backend = "cpu"` (reference, simpler).
  - `solver_backend = "cuda"` (future optimization).

### 6.2 Blockwise Solve Pipeline

For each weight matrix `W`:

1. For each row `oc`:
   - For each group `g`:
     1. Get `w_g` (`[G]`) and `H_g` (`[G, G]`).
     2. Compute GPTQ solution:
        - Determine quantizer (scale, zero).
        - Iterative rounding of `w_g` to 4-bit with Hessian-weighted error updates.
     3. Output:
        - `q_g` (`INT4[G]`) and scale (scalar or per-block; v1 uses per-block).
2. Feed `q_g` + scale into **Forge Core** packer (next section).

### 6.3 Integration with Existing GPTQ Code

v1 can:

- Initially reuse known GPTQ Python solvers for reference (e.g., simplified adaptation from public GPTQ implementations).
- Wrap them in a consistent interface:

  - `solve_block(H_g, w_g, bit_width=4, group_size=128) -> (q_g, scale_g, zero_g)`

Later, custom CUDA/Marlin-style solvers can replace the Python reference while preserving this interface.

---

## 7. Quantization & Packing Kernels (“Forge Core”)

These CUDA kernels take the **solved** quantization (or raw weights + H if we fuse) and produce backend-ready packed weights.

### 7.1 Baseline Kernel Interface

Core kernel(s) exported via a Torch extension:

- `forge_quant_pack(W, H_stats, group_size, layout_kind) -> (qweight, scales, zeros)`

v1 split:

- **Kernel A**: Pure packer
  - Inputs: per-block `q_g` (already quantized int4), `scales`, `zeros`.
  - Output: packed `qweight` + packed `zeros` in the layout expected by one backend (e.g., Marlin).

- **Kernel B**: Fused quantize + pack (optional in v1)
  - Starts from FP16 `W` and `H_g` and directly produces packed representations.

### 7.2 Layouts

For each backend:

- **Marlin W4A16 Layout**:
  - Likely requires:
    - Row-major pack over input channels.
    - Fixed nibble ordering (e.g., 8 int4 per uint32).
  - 4BIT FORGE adopts this as the **canonical v1 layout** unless overridden.

- **Generic GEMM Layout**:
  - `qweight` as `[OC, ceil(IC / 8)]` int32.
  - Scale as `[OC, num_groups]`.
  - Zero-points optional.

Kernels are:

- Warp-tiled.
- Use vectorized loads/stores (`float4`, `int4`).
- Use warp-cooperative packing (butterfly-style) for nibble placement if needed.

---

## 8. Frontend & Orchestrator

Python-level driver logic that ties all the pieces together.

### 8.1 High-Level API

Command-line or Python API like:

- `forge.calibrate(model, calib_dataset, config) -> Stats`
- `forge.quantize(model, stats, config) -> quantized_checkpoint`
- `forge.export_to_vllm(quantized_checkpoint, layout_spec)`
- `forge.export_to_compressed_tensors(quantized_checkpoint, path)`

`config` includes:

- `bits=4`
- `group_size=128`
- `num_calibration_tokens`
- `layers_to_quantize`
- `backend_layout = "marlin" | "generic" | "vllm_gptq"`

### 8.2 Control Flows

#### 8.2.1 Calibration Flow

1. Load model via I/O subsystem.
2. Build `LayerDesc`/`WeightDesc` registry.
3. Set model to eval, `torch.no_grad()`.
4. For each calibration batch:
   - Run forward pass.
   - For each hooked linear:
     - Pass activations to Statistics Engine.
5. After all batches:
   - Finalize `H_blocks` (and `g_blocks` if used).
   - Optionally move stats to CPU.
   - Save stats to disk if desired.

#### 8.2.2 Quantization Flow

1. Load calibration stats (`H_blocks`) for each weight.
2. For each layer:
   - For each quantizable `WeightDesc`:
     - Grab FP16 weights `W`.
     - For each row + block:
       - Call GPTQ Core solver to get `q_g`, `scale_g`, `zero_g`.
     - Call Forge Core packer to build backend layout `qweight`, `scales`, `zeros`.
3. Replace weights in a clone of the original model, or build a separate **quantized checkpoint** object.

#### 8.2.3 Export & Runtime Integration

1. Use `compressed-tensors` or a similar format to persist `qweight`, `scales`, and `zeros`.
2. Provide helper code to instantiate a runtime model:
   - Option A: patch `transformers` modules to use custom quantized Linear layers.
   - Option B: export weights directly into a format that `vllm` GPTQ backends can consume.

---

## 9. Invariants and Sanity Checks

- All `H_blocks` must be finite (no NaNs/inf) and positive semi-definite up to numerical noise.
- `group_size` must divide `IC` or the padding behavior must be explicitly handled.
- For each quantized weight:
  - The dequantized `W_hat` should be numerically reasonable compared to original `W`:
    - MSE and max error metrics logged per layer.
- Packed layouts must be internally consistent:
  - Round-trip test:
    - `packed -> dequantize -> fp16` must match a reference implementation within tolerance.
- Calibration is deterministic given:
  - Fixed dataset, seeds, and number of tokens.

---

## 10. Roadmap Hooks (Non-v1)

These are **not** part of v1, but the spec is written so they fit later:

- AWQ backend (alternative quantization core).
- MoE-aware pruning and expert maps.
- Multi-GPU calibration and quantization.
- Mixed-backend exports (e.g. vLLM + Triton custom kernels).
- Advanced Hessian compression (e.g. SVD per block) for extremely large models.

For now, 4BIT FORGE v1 is “just” a **clean, kernel-accelerated GPTQ pipeline** that sits on top of existing libraries and produces high-quality W4A16-G128 quantized checkpoints for large LLMs.
