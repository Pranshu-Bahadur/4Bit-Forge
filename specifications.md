# 4BIT FORGE: GPTQ Implementation Specification & Kernel Reference

**Version:** 2.1 (GPTQ, activation-iterator Hessian)
**Target:** DeepSeek-scale models (300B–671B) → 4-bit GPTQ W4A16
**Goal:** Minimal, kernel-centric toolkit for GPTQ quantization + runtime kernels.

---

## 0. Global Design

### 0.1 Objective

4Bit Forge should:

* Take a **HuggingFace / vLLM-style checkpoint folder** (sharded safetensors or `.bin`).
* Run **GPTQ 4-bit weight quantization (W4A16)** per linear layer.
* Emit a **sharded quantized checkpoint** with clearly defined layout for:

  * Custom W4A16 matmul kernels (`group_gemm` / “marlin-style”).
  * Future vLLM / custom runtime adapters.

### 0.2 Core Principles

* **Minimal Python, maximal CUDA.**

  * Python is just glue + orchestration.
  * All heavy math goes into kernels or a small number of core Torch routines.
* **Layer-local GPTQ.**

  * Calibration is per layer; Hessian is built from that layer’s input activations only.
  * Quantization consumes:

    * The layer’s full-precision weight `W`.
    * Groupwise Hessian blocks `H_blocks` (block-diagonal approximation).
* **Activation-agnostic Hessian core.**

  * Hessian builder only sees an **iterator of activations**.
  * It doesn’t care *how* those activations were produced (full model, subgraph, offloaded tensors).

---

## 1. Repo Layout (Minimal Files)

Keep the structure small and focused:

```text
4bit-forge/
  forge/
    __init__.py
    io.py          # config + streaming weights + sharded save
    gptq.py        # Hessian core + GPTQ orchestration (minimal Python)
    kernels.py     # C++/CUDA bindings only (no logic)
  csrc/
    forge_kernels.cu    # all CUDA kernels + pybind shim
  tests/
    test_gptq_hessian.py    # Hessian parity vs dense
    test_gptq_small.py      # small-layer GPTQ correctness
    test_layout_parity.py   # pack/unpack correctness
    test_group_gemm.py      # runtime GEMM correctness vs FP
```

You can merge tests later if you’re chasing LOC, but structurally this is enough.

---

## 2. I/O & Config (`forge/io.py`)

Goal: solve all checkpoint + config I/O in **~200–300 lines**.

### 2.1 Config Loader

```python
from transformers import AutoConfig

def load_model_config(model_dir: str, trust_remote_code: bool = True):
    """
    Thin wrapper around AutoConfig.from_pretrained(model_dir).
    """

def extract_llm_dims(config) -> dict:
    """
    Returns:
        {
          "hidden_size": int,
          "num_attention_heads": int,
          "num_hidden_layers": int,
          "num_key_value_heads": Optional[int],
          ...
        }
    """
```

Use this for sanity checks and for sizing matmul / buffers when needed.

### 2.2 Streaming Weight Loader (Input)

Single high-level entrypoint:

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
    """
```

Implementation sketch:

* If `model.safetensors.index.json` exists:

  * Load JSON.
  * Group `weight_map` by shard file.
  * For each shard:

    * `safe_open(..., framework="pt", device="cpu")`
    * Yield `(name, tensor)` for all params in that file.
* Else if `.bin` index exists:

  * Same idea with `torch.load` on each shard.
* Else:

  * Try `model.safetensors` or `pytorch_model.bin` directly.

`allowed_prefixes` lets you ignore optimizer states, extra heads, etc.

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

    weight_map[name] = shard_filename
    metadata stored in top-level "metadata" field.
    """
```

Logic:

* Compute `.numel() * element_size()` per tensor.
* Greedy pack tensors into shards under `max_shard_size_bytes`.
* Use `safetensors.torch.save_file` per shard.
* Build and write `forge_model.safetensors.index.json`.

### 2.4 Activation Offload (Optional, Tiny)

For large calibration runs, you can offload layer inputs:

```python
def save_activation(layer_idx: int, batch_idx: int, x: torch.Tensor, offload_dir: str) -> None:
    # torch.save({"hidden_states": x.cpu()}, f"{offload_dir}/layer{layer_idx}_batch{batch_idx}.pt")

def activation_stream_from_files(
    paths: list[str],
    key: str = "hidden_states",
):
    for p in paths:
        obj = torch.load(p, map_location="cpu")
        yield obj[key] if isinstance(obj, dict) else obj
```

Small, reusable, and keeps main logic clean.

---

## 3. GPTQ Calibration & Hessian (`forge/gptq.py`)

### 3.1 Activation Sources (Pluggable)

We treat activations (inputs to a given linear) as coming from **an iterator**:

```python
from typing import Iterable, Generator, Dict
import torch
import torch.nn as nn

def activation_stream_from_model_layer(
    model: nn.Module,
    layer: nn.Linear,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Generator[torch.Tensor, None, None]:
    """
    Yields layer-input activations for calibration.
    Shapes: [B, S, D] or [T, D].
    """
    # full model path (hook-based)
    ...

def activation_stream_from_files(
    paths: list[str],
    key: str = "hidden_states",
):
    """
    Yields layer-input activations loaded from disk.
    """
    ...
```

These are **low-LOC helper functions**, not central logic.

### 3.2 Core Hessian Builder (Pure Torch)

This is the core we already wrote; spec it formally:

```python
@torch.no_grad()
def accumulate_groupwise_hessian_from_activations(
    activation_iter: Iterable[torch.Tensor],
    in_features: int,
    group_size: int = 128,
    max_tokens: int = 4096,
    device: torch.device = torch.device("cuda"),
    act_dtype: torch.dtype = torch.float16,
    hess_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, int]:
    """
    Core GPTQ Hessian builder for one layer.

    Inputs:
      activation_iter: iterable yielding layer inputs:
                       - [T, D] or [B, S, D], last dim = in_features.
      in_features    : D.
      group_size     : GPTQ block size along input dim (multiple of 32).
      max_tokens     : cap on total tokens used.
      device         : where to accumulate H.
      act_dtype      : dtype for activations during accumulation.
      hess_dtype     : dtype for H blocks (usually fp32).

    Output:
      H_blocks: [G, group_size, group_size], where
                G = ceil(D / group_size), last block padded with zeros.
      n_seen  : number of tokens actually used.
    """
```

Mathematically:

* Let `X ∈ R^{N × D}` be concatenation of all activations.

* For each group `g` of input channels:

  * Indices `[g * group_size : g * group_size + d_g]`
  * Block Hessian:

    [
    H_g = \frac{1}{N} X_g^\top X_g \in \mathbb{R}^{d_g \times d_g}
    ]

* We store `H_blocks[g, :d_g, :d_g] = H_g`, rest zero.

This is the **canonical GPTQ Hessian representation** for Forge.

### 3.3 Optional GPU Hessian Kernel

For very large `D` and large `max_tokens`, a CUDA version can replace the pure Torch loop.

Interface (Python wrapper):

```python
def gptq_build_hessian_cuda(
    activation_tensor: torch.Tensor,    # [T, D] on CUDA
    group_size: int,
) -> torch.Tensor:                      # [G, group_size, group_size] on CUDA
    """
    CUDA kernel path; mirrors accumulate_groupwise_hessian_from_activations
    but for a single big activation tensor.
    """
```

You can keep this optional and fall back to the pure Torch version.

---

## 4. GPTQ Quantization for One Linear Layer (`forge/gptq.py`)

### 4.1 GPTQ Input/Output Contract

Given:

* Weight `W`: `[out_features, in_features]` (FP16/FP32).
* Groupwise Hessian `H_blocks`: `[G, group_size, group_size]`.
* GPTQ settings:

  * `block_size` (input grouping).
  * `symmetric` vs `asymmetric` (zero-points).

We want:

* Unpacked quantized weights `qweight_u4`: `[out_features, in_features]` int8 (low 4 bits meaningful).
* Scale and zero tensors, per group and output:

  * `scales`: `[(in_groups), out_features]` (or `[G, out]`), FP16.
  * `qzeros`: `[(in_groups), out_features]` (or packed later).

### 4.2 Python API (Orchestrator Level)

Keep the Python side **very thin**:

```python
def gptq_quantize_linear_from_H(
    weight: torch.Tensor,         # [out, in], fp16/fp32, CUDA
    H_blocks: torch.Tensor,       # [G, block_size, block_size], CUDA fp32
    block_size: int = 128,
    symmetric: bool = True,
    use_cuda_kernels: bool = True,
):
    """
    Returns:
      qweight_u4: [out, in] int8 (low 4 bits used)
      scales:     [G, out] or [in_groups, out]
      qzeros:     [G, out] or [in_groups, out] (unpacked or to-be-packed)
    """
```

Internally:

* If `use_cuda_kernels`:

  * Call `gptq_quantize_rows_launch` (CUDA kernel).
* Else:

  * Fallback to CPU/reference GPTQ (slow, only for tests).

You can also add a helper that wraps Hessian building:

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
    Convenience path:
      1. H_blocks = accumulate_groupwise_hessian_from_activations(...)
      2. gptq_quantize_linear_from_H(...)
    """
```

But the **core contract** is: GPTQ quantization operates on `(W, H_blocks)`.

---

## 5. CUDA Kernels (`csrc/forge_kernels.cu`)

All kernels live in **one** CU file with pybind at the bottom.

### 5.1 Common Definitions

* `#define WARP_SIZE 32`

* 4-bit representation:

  * Store as signed 4-bit two’s complement in low nibble of `int8` during intermediate steps.
  * Packed into `uint32_t` with 8 nibbles per word.

* Grouping:

  * `group_size` (a.k.a. block size) ∈ {128, 256, 512}, multiple of 32.
  * `G = ceil(D / group_size)` groups along input dim.

### 5.2 (Optional) Groupwise Hessian Kernel

You may implement a GPU Hessian builder later; spec:

```cpp
void gptq_build_hessian_launch(
    torch::Tensor inputs,     // [T, D], float16/float32, CUDA
    torch::Tensor H_blocks,   // [G, group_size, group_size], float32, CUDA
    int group_size
);
```

Kernel logic per group `g`:

* Load `X_g` = inputs[:, start:end] into registers/SMEM tiles.
* Accumulate `H_g += X_g^T X_g`.
* Normalize by T at the end.

This is effectively the CUDA version of `accumulate_groupwise_hessian_from_activations` when you have a big `[T, D]` tensor on device.

### 5.3 GPTQ Row Quantization Kernel (Core)

This is the main GPTQ engine on GPU.

Launcher:

```cpp
void gptq_quantize_rows_launch(
    torch::Tensor weight,      // [out, in], float16/float32, CUDA
    torch::Tensor H_blocks,    // [G, group_size, group_size], float32, CUDA
    torch::Tensor qweight_u4,  // [out, in], int8, CUDA (low 4 bits used)
    torch::Tensor scales,      // [G, out], float16
    torch::Tensor qzeros,      // [G, out], int8 or int32 (unpacked)
    int block_size,
    bool symmetric
);
```

Per row `out_idx`:

1. For each group `g`:

   * Slice `w_block` = `w[out_idx, start:end]`.
   * Slice Hessian block `H_g`.
   * Apply GPTQ update:

     * Compute factorization `H_g = L L^T` or approximate.
     * Classic GPTQ step: quantize and adjust residual using H-norm metric.
   * Compute scale (and optionally zero) for that block.
   * Write quantized 4-bit values (as `int8` low nibble) into `qweight_u4[out_idx, start:end]`.
   * Write scale/zero into `scales[g, out_idx]`, `qzeros[g, out_idx]`.

For v1, you can start with a simpler per-block scheme:

* Scale = max-abs / 7.
* Zero-point = 0 (symmetric).
* Use H_blocks only to weight errors or for later iterations.

The full GPTQ search / triangular solve can be added once the structure is stable.

### 5.4 Int4 Packing Kernel

Purpose: pack `qweight_u4` into a matmul-friendly layout.

Launcher:

```cpp
void pack_int4_launch(
    torch::Tensor qweight_u4,     // [out, in], int8 (low 4 bits)
    torch::Tensor qweight_packed, // [in, out_packs] or [out, in_packs], int32
    bool transpose_for_runtime
);
```

Layout choice (one canonical version):

* **Runtime layout:** `[IC, OC_packs]`, where `OC_packs = ceil(OC / 8)`.

  * `ic` = input channel index.
  * `oc_pack` = block of 8 consecutive output channels.

* For channels `c..c+7` and fixed `ic`, you have 8 weights:

  ```text
  q0 = w_q[c + 0, ic]
  q1 = w_q[c + 1, ic]
  ...
  q7 = w_q[c + 7, ic]
  ```

* Packed into an `uint32_t` with nibble order chosen for your matmul kernel.
  Example (vLLM-style):

  ```text
  Bits 0–3  <- q0
  Bits 4–7  <- q2
  Bits 8–11 <- q4
  Bits 12–15<- q6
  Bits 16–19<- q1
  Bits 20–23<- q3
  Bits 24–27<- q5
  Bits 28–31<- q7
  ```

Kernel details:

* One warp handles a tile, using a **butterfly transpose** if you need to rotate data into packing order without SMEM bank conflicts.
* You already have PackBoost-style templates (`encode_cuts` / butterfly transpose) to emulate.

### 5.5 group_gemm / “Marlin-Style” W4A16 Matmul

Runtime kernel, separate from quantization:

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

* For each tile `[B_tile, out_tile]`:

  * For each group `g` of input channels:

    * Load a chunk of packed `qweight` and `qzeros`.

    * Unpack 4-bit nibbles into registers.

    * Dequantize on the fly:

      [
      w_{\text{fp}} = (q - z) \cdot \text{scale}
      ]

    * Multiply with corresponding `x` slice and accumulate into `y`.

* Marlin / group_gemm techniques applied:

  * Vectorized loads (`int4`/`int8`/`uint32_t`).
  * Shared memory tiling.
  * Per-warp MMA loops.

This can be iterated on separately without touching GPTQ code.

---

## 6. Python Bindings (`forge/kernels.py`)

This file stays tiny:

```python
import torch
from torch.utils.cpp_extension import load

_forge_kernels = load(
    name="forge_kernels",
    sources=["csrc/forge_kernels.cu"],
    extra_cuda_cflags=[...],
    extra_cflags=[...],
)

gptq_build_hessian   = getattr(_forge_kernels, "gptq_build_hessian", None)
gptq_quantize_rows   = _forge_kernels.gptq_quantize_rows
pack_int4            = _forge_kernels.pack_int4
group_gemm_w4a16     = _forge_kernels.group_gemm_w4a16
```

No logic, just exports.

---

## 7. Tests (`tests/`)

Focus on *sharp* tests that exercise the math and layouts.

### 7.1 `test_gptq_hessian.py`

* Construct random activations `X ∈ R^{T×D}`.

* Call:

  ```python
  H_blocks, n_seen = accumulate_groupwise_hessian_from_activations(
      activation_iter=[X],
      in_features=D,
      group_size=group_size,
      max_tokens=T,
      device=device,
      act_dtype=torch.float32,
      hess_dtype=torch.float32,
  )
  ```

* Compare to dense reference:

  ```python
  H_full = (X.T @ X) / T
  ```

* For each block `g`, verify `H_blocks[g]` matches `H_full[slice, slice]` within tolerance.

If you later implement `gptq_build_hessian_launch`, add a CUDA path and test equality with the same reference.

### 7.2 `test_gptq_small.py`

* Tiny linear:

  ```python
  W = torch.randn(16, 32, dtype=torch.float32, device=device)
  X = torch.randn(256, 32, dtype=torch.float32, device=device)
  ```

* Build `H_blocks` via `accumulate_groupwise_hessian_from_activations`.

* Run `gptq_quantize_linear_from_H` → `qweight_u4, scales, qzeros`.

* Dequantize:

  ```python
  W_q_approx = dequantize(qweight_u4, scales, qzeros, block_size)
  ```

* Compare `X @ W.T` vs `X @ W_q_approx.T`.

### 7.3 `test_layout_parity.py`

* Random int4 tensor:

  ```python
  q_u4 = torch.randint(-8, 8, (out, in_), dtype=torch.int8)
  ```

* Pack → `q_packed`.

* Unpack via CPU reference.

* Assert equality of all 4-bit values.

### 7.4 `test_group_gemm.py`

* Setup:

  ```python
  B, D, O = 4, 64, 32
  X = torch.randn(B, D, dtype=torch.float16, device=device)
  W = torch.randn(O, D, dtype=torch.float32, device=device)
  ```

* Quantize `W` → `qweight_packed, scales, qzeros`.

* Run `group_gemm_w4a16`.

* Compare result to `X @ W.T` (FP32 baseline) within reasonable tolerance.

---

## 8. End-to-End Orchestrator (Quantize Whole Model)

Simple driver sketch:

```python
from forge.io import unified_weights_iterator, save_sharded_safetensors, load_model_config
from forge.gptq import (
    accumulate_groupwise_hessian_from_activations,
    gptq_quantize_linear_from_H,
)

def forge_gptq_quantize_model(
    model_dir: str,
    calib_activations_resolver,   # function that yields activation_iter for given (layer_idx)
    output_dir: str,
    block_size: int = 128,
    max_tokens: int = 4096,
):
    cfg = load_model_config(model_dir)
    quant_state = {}

    for name, w_cpu in unified_weights_iterator(model_dir, allowed_prefixes=["model."]):
        if is_linear_weight_you_care_about(name, w_cpu):
            layer_idx, proj_kind = parse_layer_and_proj(name)

            activation_iter = calib_activations_resolver(layer_idx)  # iterable
            H_blocks, _ = accumulate_groupwise_hessian_from_activations(
                activation_iter=activation_iter,
                in_features=w_cpu.shape[1],
                group_size=block_size,
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

            # Layout your final names however you want:
            quant_state[f"{name}.qweight_u4"] = qweight_u4.cpu()
            quant_state[f"{name}.scales"]    = scales.cpu()
            quant_state[f"{name}.qzeros"]    = qzeros.cpu()
        else:
            quant_state[name] = w_cpu  # copy unchanged

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

The calibrations source (`calib_activations_resolver`) can be:

* Hook-based full-model forward.
* Offloaded safetensors on disk.
* Synthetic X for quick experiments.
