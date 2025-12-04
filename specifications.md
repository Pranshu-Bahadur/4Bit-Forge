# 4BIT FORGE: GPTQ Implementation Specification & Kernel Reference

**Version:** 2.2 (GPTQ, Streaming Calibration, MoE-Aware)  
**Target:** DeepSeek-scale models (300B–671B, MoE) → 4-bit GPTQ W4A16  
**Goal:** Minimal, kernel-centric toolkit for GPTQ quantization + runtime kernels, with
          streaming-friendly calibration and per-expert MoE support.

---

## 0. Global Design

### 0.1 Objective

4Bit Forge should:

- Take a **HuggingFace / vLLM-style checkpoint folder** (sharded safetensors or `.bin`).
- Run **GPTQ 4-bit weight quantization (W4A16)** per *linear*:
  - Dense linears (attention/MLP).
  - MoE experts’ linears (per expert).
- Emit a **sharded quantized checkpoint** with a clear layout for:
  - Custom W4A16 matmul kernels (`group_gemm` / “Marlin-style”).
  - Future vLLM / custom runtime adapters.

### 0.2 Core Principles

- **Minimal Python, maximal CUDA.**
  - Python = glue + orchestration + reference paths.
  - Heavy math = CUDA kernels, small number of core Torch ops.
- **Layer-local GPTQ.**
  - Calibration is per layer / per expert.
  - Hessian for a layer/expert only uses its own input activations.
- **Streaming-first.**
  - Calibration does **not** assume the full model fits in GPU VRAM.
  - Both weights and activations are treated as streams:
    - Weights: safetensors or `.bin` shards, loaded lazily.
    - Activations: batch-by-batch, layer-by-layer, offloaded if needed.
- **MoE-aware.**
  - MoE layers are treated as:
    - A gate + multiple experts.
    - Each expert has its own Hessian and GPTQ quantization.

---

## 1. Repo Layout (Minimal Files)

Keep the structure small and focused:

```text
4bit-forge/
  forge/
    __init__.py
    io.py          # config + streaming weights + sharded save + simple activation I/O
    gptq.py        # Hessian core + GPTQ orchestration (minimal Python)
    kernels.py     # C++/CUDA bindings only (no logic)
  csrc/
    forge_kernels.cu    # all CUDA kernels + small C++/pybind shim
  tests/
    test_gptq_hessian.py    # Hessian parity vs dense (Torch reference)
    test_gptq_small.py      # small-layer GPTQ correctness
    test_layout_parity.py   # pack/unpack correctness
    test_group_gemm.py      # runtime GEMM correctness vs FP
````

---

## 2. I/O & Config (`forge/io.py`)

### 2.1 Config Loader

```python
from transformers import AutoConfig, PretrainedConfig

def load_model_config(model_dir: str, trust_remote_code: bool = True) -> PretrainedConfig:
    """Thin wrapper around AutoConfig.from_pretrained(model_dir)."""

def extract_llm_dims(config: PretrainedConfig) -> dict:
    """
    Returns a dict like:
        {
          "hidden_size": int,
          "num_attention_heads": int,
          "num_hidden_layers": int,
          "num_key_value_heads": Optional[int],
          "intermediate_size": Optional[int],
          "num_experts": Optional[int],            # for MoE
          "num_experts_per_tok": Optional[int],    # for MoE gating
          ...
        }
    """
```

Used for sanity checks and sizing matmul/buffers.

### 2.2 Streaming Weight Loader (Input)

High-level entrypoint:

```python
from typing import Iterator, Tuple
import torch

TensorIter = Iterator[Tuple[str, torch.Tensor]]

def unified_weights_iterator(
    model_dir: str,
    allowed_prefixes: list[str] | None = None,
) -> TensorIter:
    """
    Yields (param_name, cpu_tensor).

    Supports:
      - Sharded safetensors with model.safetensors.index.json
      - Sharded bin with pytorch_model.bin.index.json
      - Single-file safetensors or bin.

    'allowed_prefixes' can filter parameters, e.g. ["model."].
    """
```

Implementation sketch:

* If `model.safetensors.index.json` exists:

  * Load the JSON, group `weight_map` by shard filename.
  * For each shard:

    * `safe_open(path, framework="pt", device="cpu")` as `f`.
    * Yield `(name, f.get_tensor(name))` for each tensor listed in that file.
    * Close file → release RAM.
* Else if `pytorch_model.bin.index.json` exists:

  * Similar logic with `torch.load` shard-by-shard.
* Else:

  * Try `model.safetensors` or `pytorch_model.bin`.

This is used both for:

* **Quantization-time weight streaming** (quantizing one layer at a time).
* Potential **streaming calibration runners** that build partial models.

### 2.3 Sharded Saver (Output)

```python
from typing import Mapping

def save_sharded_safetensors(
    state_dict: Mapping[str, torch.Tensor],
    output_dir: str,
    max_shard_size_gb: float = 5.0,
    metadata: dict | None = None,
) -> None:
    """
    Writes:
      - forge_model-00001-of-000NN.safetensors
      - ...
      - forge_model.safetensors.index.json

    The index file has:
      {
        "metadata": { ... },
        "weight_map": { param_name: shard_filename, ... }
      }
    """
```

Logic:

* Compute bytes per tensor: `tensor.numel() * tensor.element_size()`.
* Greedy pack into shards ≤ `max_shard_size`.
* For each shard:

  * `safetensors.torch.save_file(sub_state_dict, shard_path, metadata=None)`.
* Build `weight_map` and write `forge_model.safetensors.index.json`.

### 2.4 Activation Offload Helpers (Optional)

For large calibration runs:

```python
import torch
from typing import Iterable, Generator, Dict

def save_activation(
    layer_key: str,          # e.g. "layer_12.w1" or "layer_12.expert_3.w2"
    batch_idx: int,
    x: torch.Tensor,
    offload_dir: str,
) -> None:
    path = f"{offload_dir}/{layer_key}_batch{batch_idx:05d}.pt"
    torch.save({"hidden_states": x.cpu()}, path)

def activation_stream_from_files(
    paths: list[str],
    key: str = "hidden_states",
) -> Generator[torch.Tensor, None, None]:
    for p in paths:
        obj = torch.load(p, map_location="cpu")
        yield obj[key] if isinstance(obj, dict) else obj
```

These are small utilities; they keep main calibration logic clean.

---

## 3. Calibration & Streaming

### 3.1 Activation Sources (Abstract)

We treat “inputs to a given linear (or expert)” as coming from an **iterator** of tensors, not from a fixed in-memory dataset.

```python
from typing import Iterable, Generator

def activation_stream_from_model_layer(
    model,
    layer_path: str,
    dataloader,
    device,
) -> Generator[torch.Tensor, None, None]:
    """
    Streams inputs to a given linear (dense or expert) by registering hooks
    on the model and running forward passes batch-by-batch.
    Yields tensors with last dim == in_features.
    """
    ...

def activation_stream_from_files(
    paths: list[str],
    key: str = "hidden_states",
):
    """
    Streams pre-saved activations from disk (see 2.4).
    """
    ...
```

The Hessian builder only requires an **Iterable[Tensor]**; how those tensors are produced is up to the calibration backend.

---

### 3.2 Streaming Calibration for Huge Models (1T-Scale)

For DeepSeek-scale MoE models that **cannot fit in a single GPU’s VRAM**, 4Bit Forge assumes a **layer-wise streaming calibrator**:

**Conceptual pipeline (per calibration pass):**

1. **Dataloader streaming**
   Iterate over calibration batches on CPU (or small GPU staging).

2. **Layer-wise streaming forward:**

   For each transformer layer index `L`:

   * Use the checkpoint + model config to know which parameters belong to layer `L`:

     * Dense: `model.layers.L.self_attn.*`, `model.layers.L.mlp.*`
     * MoE:   `model.layers.L.experts.[0..E-1].*`, gate weights.

   * For a given batch:

     1. Load only the weights for layer `L` from safetensors shards (via `unified_weights_iterator` filtered by name).
     2. Move those weights to the GPU.
     3. Feed the layer’s input hidden states through this layer to obtain:

        * new hidden states for the next layer.
        * activations that are inputs to the target linears (dense or experts).
     4. For each target linear / expert:

        * Feed its input activations into `accumulate_groupwise_hessian_*` (see below).
     5. Drop GPU-resident weights for layer `L`; keep only Hessian accumulators (small) on device.

   * Move to `L+1`, reusing the updated hidden states as input.

3. **MoE gating in streaming mode:**

   * Gating decisions happen during the (`L`) forward pass.
   * Each token is routed to one or more experts.
   * In practice, you:

     * Collect **routed activations per expert** inside the MoE layer (via hooks).
     * For each expert `e`:

       * Accumulate Hessian with its own activation stream.

4. **Memory model:**

   * At any point:

     * Only one or a few layers’ weights are in GPU memory.
     * Hidden states are kept as `[batch, seq, hidden_size]` for the current calibration step.
   * No assumption that the full model fits in VRAM.

The exact implementation details of this streaming calibrator are outside Forge core; the core just needs an `activation_iter` per (layer, expert).

---

### 3.3 Dense Layer Hessian (Core Function)

Core “single-layer, dense” Hessian builder:

```python
import torch
from typing import Iterable, Tuple

@torch.no_grad()
def accumulate_groupwise_hessian_from_activations_tensor(
    inputs: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """
    Convenience reference:

    inputs: [N, in_features] on CUDA or CPU
    block_size: int (e.g. 128) along input dim.

    Returns:
      H_blocks: [n_blocks, block_size, block_size] float32
                with last block padded as needed.

    Definition:
      Let D = in_features, G = ceil(D / block_size).
      For each group g:
        start = g * block_size
        end   = min(start + block_size, D)
        Xg    = inputs[:, start:end]
        Hg    = (Xg.T @ Xg) / N
      H_blocks[g, :d, :d] = Hg, with d = end - start.
    """
```

Streaming/iterator version:

```python
@torch.no_grad()
def accumulate_groupwise_hessian_from_activations_iter(
    activation_iter: Iterable[torch.Tensor],
    in_features: int,
    block_size: int = 128,
    max_tokens: int = 4096,
    device: torch.device = torch.device("cuda"),
    act_dtype: torch.dtype = torch.float16,
    hess_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, int]:
    """
    Streaming Hessian builder for one layer.

    Inputs:
      activation_iter: iterable yielding layer inputs; shapes can be:
                        - [T, D]
                        - [B, S, D]
                       (last dim must equal in_features)
      in_features    : D
      block_size     : grouping along D (GPTQ block size)
      max_tokens     : cap on total tokens used for H (for calibration budget)
      device         : where to keep H_blocks
      act_dtype      : cast inputs to this before accumulation
      hess_dtype     : dtype of H_blocks

    Output:
      H_blocks: [G, block_size, block_size] on 'device'
      n_seen  : number of tokens actually accumulated (<= max_tokens)
    """
```

This is the canonical layer Hessian representation used by GPTQ in Forge.

---

### 3.4 MoE Layer Hessian (Per Expert)

For a MoE layer with `E` experts, each expert `e` has its own linear(s) and its own Hessian:

```python
from typing import Dict

@torch.no_grad()
def accumulate_moe_hessians(
    moe_activation_iter: Iterable[Dict[int, torch.Tensor]],
    in_features: int,
    block_size: int = 128,
    max_tokens_per_expert: int = 4096,
    device: torch.device = torch.device("cuda"),
) -> Dict[int, Tuple[torch.Tensor, int]]:
    """
    MoE Hessian accumulation.

    moe_activation_iter yields per-batch dicts:

      {
        expert_id_0: X0,   # [N0, D]
        expert_id_1: X1,   # [N1, D]
        ...
      }

    For each expert 'e', we maintain its own:
      H_blocks[e]: [G, block_size, block_size]
      n_seen[e]   : number of tokens for expert e

    Returns:
      {
        expert_id: (H_blocks_e, n_seen_e)
      }
    """
```

How `moe_activation_iter` is produced:

* Inside a streaming forward of a MoE layer:

  * Use the gate outputs / routing indices to bucket tokens by expert.
  * For each batch, build a dict `expert_id -> expert_inputs` and yield it.

The Hessian math is unchanged; only the **routing** differs.

---

## 4. GPTQ Quantization for One Linear Layer (`forge/gptq.py`)

### 4.1 GPTQ Input/Output Contract

For **one linear** with weight `W ∈ R[out, in]`:

Inputs:

* `weight`: `[out_features, in_features]` (FP16/FP32, CUDA).
* `H_blocks`: `[G, block_size, block_size]` (FP32, CUDA).
* GPTQ config:

  * `block_size` == `group_size` along input dim.
  * `symmetric` (zero-point = 0) vs future asymmetric.

Outputs:

* `qweight_u4`: `[out_features, in_features]` int8 (low 4 bits used).
* `scales`:     `[G, out_features]` FP16 (or FP32).
* `qzeros`:     `[G, out_features]` int8 or int32 (unpacked zero-points).

### 4.2 Python API: From Hessian

```python
import torch
from typing import Tuple

def gptq_quantize_linear_from_H(
    weight: torch.Tensor,         # [out, in], fp16/fp32, CUDA
    H_blocks: torch.Tensor,       # [G, block_size, block_size], fp32, CUDA
    block_size: int = 128,
    symmetric: bool = True,
    use_cuda_kernels: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single-layer GPTQ quantization using precomputed H_blocks.

    Returns:
      qweight_u4: [out, in], int8 (low 4 bits used)
      scales:     [G, out]
      qzeros:     [G, out] (unpacked; may be all zeros if symmetric)
    """
```

Implementation:

* If `use_cuda_kernels`:

  * Calls `gptq_quantize_rows_launch` (CUDA).
* Else:

  * CPU/Torch reference path for testing only.

### 4.3 Convenience: From Activations (Dense or MoE)

Dense per-layer:

```python
def gptq_quantize_linear_from_activations(
    weight: torch.Tensor,
    activation_iter,
    in_features: int,
    block_size: int = 128,
    max_tokens: int = 4096,
    symmetric: bool = True,
    device: torch.device = torch.device("cuda"),
):
    """
    1. H_blocks = accumulate_groupwise_hessian_from_activations_iter(...)
    2. Run gptq_quantize_linear_from_H(...)
    """
```

MoE per-expert will typically be orchestrated outside this helper:

* For each expert `e`:

  * Build `H_blocks_e`.
  * Call `gptq_quantize_linear_from_H(W_e, H_blocks_e, ...)`.

---

## 5. CUDA Kernels (`csrc/forge_kernels.cu`)

All kernels live in one compilation unit with a `PYBIND11_MODULE` shim.

### 5.1 Common Definitions

* `#define WARP_SIZE 32`
* 4-bit representation:

  * Intermediate: `int8` with low nibble ∈ `[-8..7]` in two’s complement.
  * Packed: `uint32_t` with 8 nibbles → 32 bits.
* Grouping:

  * `group_size` (GPTQ block size) ∈ {128, 256, 512}, multiple of 32.
  * `G = ceil(D / group_size)`.

### 5.2 (Optional) Groupwise Hessian Kernel

Interface:

```cpp
void gptq_build_hessian_launch(
    torch::Tensor inputs,     // [T, D], float16/float32, CUDA
    torch::Tensor H_blocks,   // [G, group_size, group_size], float32, CUDA
    int group_size
);
```

* Each block processes a (group, tile_of_T).
* Accumulate `H_g += X_g^T X_g` in FP32.
* Normalize by T at the end.

This is a faster alternative to `accumulate_groupwise_hessian_from_activations_tensor` when you already have `[T, D]` on device.

### 5.3 GPTQ Row Quantization Kernel

Main GPTQ engine:

```cpp
void gptq_quantize_rows_launch(
    torch::Tensor weight,      // [out, in], float16/float32, CUDA
    torch::Tensor H_blocks,    // [G, group_size, group_size], float32, CUDA
    torch::Tensor qweight_u4,  // [out, in], int8, CUDA (low 4 bits used)
    torch::Tensor scales,      // [G, out], float16
    torch::Tensor qzeros,      // [G, out], int8 or int32, CUDA
    int block_size,
    bool symmetric
);
```

Per row `out_idx`:

* For each group `g` along input dim:

  * Slice `w_block = weight[out_idx, start:end]`.
  * Get `H_g` = `H_blocks[g]`.
  * GPTQ step for that block:

    * Factorize `H_g` (e.g. Cholesky).
    * Use H-norm–aware quantization to choose 4-bit values.
    * Compute scale (and zero-point if asymmetric).
  * Write:

    * `qweight_u4[out_idx, start:end]` (low nibble).
    * `scales[g, out_idx]`, `qzeros[g, out_idx]`.

For v1, a simplified scheme is allowed:

* `scale_g = max_abs(w_block) / 7`.
* `q = clamp(round(w_block / scale_g), -8, 7)`.
* `zero = 0` if `symmetric`.
* Still store `H_blocks` for future upgrades / debugging.

### 5.4 Int4 Packing Kernel

Packs `qweight_u4` into a layout for matmul kernels.

```cpp
void pack_int4_launch(
    torch::Tensor qweight_u4,     // [out, in], int8 (low 4 bits)
    torch::Tensor qweight_packed, // e.g. [in, out_packs], int32
    bool transpose_for_runtime
);
```

Canonical runtime layout (example):

* `qweight_packed`: `[IC, OC_packs]`, with `OC_packs = ceil(OC / 8)`.

Packing:

* For each `ic` and each `oc_pack`:

  * Let `c = oc_pack * 8`.

  * Read:

    ```text
    q0 = qweight_u4[c + 0, ic]
    q1 = qweight_u4[c + 1, ic]
    ...
    q7 = qweight_u4[c + 7, ic]
    ```

  * Pack into an `uint32_t` with a chosen nibble order (e.g. vLLM-like):

    ```text
    Bits 0–3   <- q0
    Bits 4–7   <- q2
    Bits 8–11  <- q4
    Bits 12–15 <- q6
    Bits 16–19 <- q1
    Bits 20–23 <- q3
    Bits 24–27 <- q5
    Bits 28–31 <- q7
    ```

* Kernel:

  * Warp per tile.
  * Use register-based or shared-memory-plus-butterfly transpose (PackBoost-style) to optimize coalesced reads/writes and avoid bank conflicts.

### 5.5 group_gemm / “Marlin-Style” W4A16 Matmul

Runtime matmul:

```cpp
void group_gemm_w4a16_launch(
    torch::Tensor x,         // [B, in], float16
    torch::Tensor qweight,   // [in, out_packs], int32 (packed W4)
    torch::Tensor scales,    // [G, out], float16
    torch::Tensor qzeros,    // [G, out_packs], int32 (packed zeros)
    torch::Tensor y,         // [B, out], float16
    int group_size
);
```

Core idea:

* For each output tile:

  * Loop over input groups.
  * For each group:

    * Load a chunk of `qweight` and `qzeros`.
    * Unpack nibbles into registers.
    * Dequantize on the fly:

      * `w_fp = (q - z) * scale`.
    * Multiply with corresponding slice of `x` and accumulate into `y`.

The exact tiling / warp-level strategy follows Marlin/group_gemm patterns and can evolve independently of GPTQ.

---

## 6. Python Bindings (`forge/kernels.py`)

Just bindings, no logic:

```python
import torch
from torch.utils.cpp_extension import load

_forge_kernels = load(
    name="forge_kernels",
    sources=["csrc/forge_kernels.cu"],
    extra_cuda_cflags=[...],
    extra_cflags=[...],
)

gptq_build_hessian = getattr(_forge_kernels, "gptq_build_hessian", None)
gptq_quantize_rows = _forge_kernels.gptq_quantize_rows
pack_int4          = _forge_kernels.pack_int4
group_gemm_w4a16   = _forge_kernels.group_gemm_w4a16
```

---

## 7. Tests (`tests/`)

### 7.1 `test_gptq_hessian.py`

* Generate random activations `X ∈ R^{T×D}`.

* Compute:

  ```python
  H_blocks, n_seen = accumulate_groupwise_hessian_from_activations_iter(
      activation_iter=[X],
      in_features=D,
      block_size=block_size,
      max_tokens=T,
      device=device,
      act_dtype=torch.float32,
      hess_dtype=torch.float32,
  )
  H_full = (X.T @ X) / T
  ```

* For each block `g`, compare `H_blocks[g, :d, :d]` vs `H_full[slice, slice]`.

If CUDA Hessian kernel exists, also test parity:

* `H_blocks_cuda = gptq_build_hessian_launch(X_cuda, ...)`.

### 7.2 `test_gptq_small.py`

* Small linear:

  ```python
  W = torch.randn(16, 32, dtype=torch.float32, device=device)
  X = torch.randn(256, 32, dtype=torch.float32, device=device)
  ```

* Build Hessian:

  ```python
  H_blocks, _ = accumulate_groupwise_hessian_from_activations_iter(
      activation_iter=[X],
      in_features=32,
      block_size=16,
      max_tokens=256,
      device=device,
  )
  ```

* GPTQ:

  ```python
  qweight_u4, scales, qzeros = gptq_quantize_linear_from_H(
      W.to(device),
      H_blocks,
      block_size=16,
      symmetric=True,
      use_cuda_kernels=True,
  )
  ```

* Dequantize and compare `X @ W.T` vs `X @ W_q.T`.

### 7.3 `test_layout_parity.py`

* Random int4 values:

  ```python
  q_u4 = torch.randint(-8, 8, (out, in_), dtype=torch.int8, device=device)
  ```

* Pack → `q_packed`.

* Unpack via CPU reference.

* Assert all 4-bit values match.

### 7.4 `test_group_gemm.py`

* Setup:

  ```python
  B, D, O = 4, 64, 32
  X = torch.randn(B, D, dtype=torch.float16, device=device)
  W = torch.randn(O, D, dtype=torch.float32, device=device)
  ```

* Run Hessian + GPTQ to obtain `qweight_packed, scales, qzeros`.

* Run `group_gemm_w4a16`.

* Compare to `X @ W.T` baseline.

### 7.5 MoE Smoke Test (Optional)

* Synthetic MoE:

  * 2 experts, simple gate that routes half tokens to each.
* Build `moe_activation_iter` that yields `expert_id -> X_e`.
* Accumulate Hessians per expert.
* GPTQ each expert’s weight separately.
* Check matmul parity per expert.

---

## 8. End-to-End Orchestrator (Dense + MoE)

High-level driver (quantization only; calibration assumed done or streamed):

```python
from forge.io import unified_weights_iterator, save_sharded_safetensors
from forge.gptq import (
    accumulate_groupwise_hessian_from_activations_iter,
    gptq_quantize_linear_from_H,
)

def forge_gptq_quantize_model(
    model_dir: str,
    calib_activations_resolver,  # fn(layer_key) -> activation_iter (dense or MoE view)
    output_dir: str,
    block_size: int = 128,
    max_tokens: int = 4096,
):
    """
    calib_activations_resolver:
      - For dense layers:
          layer_key -> activation_iter (yielding [*, D])
      - For MoE experts:
          layer_key -> moe_activation_iter (yielding {expert_id: [*, D]})
    """
    quant_state = {}

    for name, w_cpu in unified_weights_iterator(model_dir, allowed_prefixes=["model."]):
        if is_linear_weight_you_care_about(name, w_cpu):
            layer_key, expert_id_opt, proj_kind = parse_layer_and_proj(name)
            in_features = w_cpu.shape[1]

            if expert_id_opt is None:
                # Dense linear
                activation_iter = calib_activations_resolver(layer_key)
                H_blocks, _ = accumulate_groupwise_hessian_from_activations_iter(
                    activation_iter=activation_iter,
                    in_features=in_features,
                    block_size=block_size,
                    max_tokens=max_tokens,
                    device=torch.device("cuda"),
                )
                qweight_u4, scales, qzeros = gptq_quantize_linear_from_H(
                    weight=w_cpu.cuda(),
                    H_blocks=H_blocks,
                    block_size=block_size,
                    symmetric=True,
                    use_cuda_kernels=True,
                )
                quant_state[f"{name}.qweight_u4"] = qweight_u4.cpu()
                quant_state[f"{name}.scales"]     = scales.cpu()
                quant_state[f"{name}.qzeros"]     = qzeros.cpu()

            else:
                # MoE expert linear
                moe_activation_iter = calib_activations_resolver(layer_key)
                # accumulate_moe_hessians returns a dict expert_id -> (H_blocks_e, n_seen_e)
                H_moe = accumulate_moe_hessians(
                    moe_activation_iter,
                    in_features=in_features,
                    block_size=block_size,
                    max_tokens_per_expert=max_tokens,
                    device=torch.device("cuda"),
                )
                H_blocks_e, _ = H_moe[expert_id_opt]
                qweight_u4, scales, qzeros = gptq_quantize_linear_from_H(
                    weight=w_cpu.cuda(),
                    H_blocks=H_blocks_e,
                    block_size=block_size,
                    symmetric=True,
                    use_cuda_kernels=True,
                )
                quant_state[f"{name}.qweight_u4"] = qweight_u4.cpu()
                quant_state[f"{name}.scales"]     = scales.cpu()
                quant_state[f"{name}.qzeros"]     = qzeros.cpu()

        else:
            # Copy untouched (e.g., embedding, layer norms, gate weights)
            quant_state[name] = w_cpu

    save_sharded_safetensors(
        quant_state,
        output_dir=output_dir,
        max_shard_size_gb=5.0,
        metadata={
            "format": "4bit-forge-gptq",
            "block_size": str(block_size),
        },
    )
```