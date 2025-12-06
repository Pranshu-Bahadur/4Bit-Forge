# 4BIT FORGE: GPTQ Core & QMeta4 Specification

**Version:** 2.4
**Focus:** qmeta4 quant grid + GPTQ solver core (MoE-Quant–style, streaming-ready design)
**Target models:** Large transformer / MoE LLMs (e.g. DeepSeek-Math-V2) with groupwise W4A16-style weight quantization.

This document describes the current design and implementation status of 4Bit Forge’s **core GPTQ engine**, centered around:

* A compact **qmeta4** format for groupwise quantization metadata.
* Fast CUDA kernels to build / refine qmeta from weights.
* A reference **GPTQ solver** that consumes qmeta + Hessian inverse and emits quantized weights.

Higher-level pieces (checkpoint I/O, calibration / Hessian builders, MoE orchestration, runtime kernels) are explicitly in scope **for the overall project**, but not implemented in this repo yet.

---

## 0. High-Level Overview

### 0.1 Goals

4Bit Forge aims to be a **minimal, composable GPTQ engine** that can be plugged into a larger quantization stack (e.g. MoE-Quant-style pipeline):

* Efficient **groupwise metadata** (scale, zero-point, flags) encoded as 4 bytes per group (**qmeta4**).
* **CUDA-accelerated grid search** to choose good per-group scales (ABSMAX or MSE/Lᵖ).
* A **row-wise GPTQ solver** that:

  * Takes a single linear weight matrix and its inverse Hessian (or factor).
  * Quantizes weights groupwise using qmeta4.
  * Propagates quantization error using the Hessian inverse (standard GPTQ logic).

Everything else (which model this layer belongs to, where Hessians come from, how you pack INT4 for matmuls) is handled by outer tooling.

### 0.2 Scope & Implementation Status

Core functionality:

* [x] **qmeta4 binary format** (C++ + Torch-side encode/decode).
* [x] **Range-based group meta builder** (CUDA + CPU reference).
* [x] **MSE / Lᵖ grid-search refinement** (CUDA + CPU reference).
* [x] **Python GPTQ API**:

  * [x] `GPTQ.build_quant_grid(...)` → qmeta4 from raw weights.
  * [x] `GPTQ.solver(...)` → GPTQ quantization given `H⁻¹` + qmeta4.
* [x] Support for fp32 / fp16 / bf16 / fp8 (E4M3) input weights in CUDA path.
* [ ] Hessian / calibration utilities (`update(input)`, `quantization_pre_step()`).
* [ ] MoE-specific helpers (expert routing, per-expert Hessians).
* [ ] Checkpoint streaming I/O helpers (safetensors shards, etc.).
* [ ] INT4 packing and W4A16 matmul runtime kernels.
* [ ] True single-GPU streaming quantization (layer-wise weight/activation streaming).

Design-wise, the core is **“streaming-ready”** and MoE-compatible: we enforce shapes / contracts that can be fed by a streaming calibration stack later, but v2.4 itself is an **offline GPTQ kernel core**.

### 0.3 “Core, Not Framework” / Non-Goals Right Now

4Bit Forge core does **not**:

* Load full LLM checkpoints (no HF/vLLM loader).
* Run end-to-end calibration or accumulate Hessians from activations.
* Implement MoE routing / expert-parallel orchestration.
* Pack INT4 or implement matmul kernels.
* Implement a full quantization CLI or training pipeline.

It is designed to be plugged into a larger stack (MoE-Quant, custom GPTQ tools, vLLM wrappers, etc.).

---

## 1. Design Principles

1. **Core, not framework (for now)**

   * This repo does *not* manage full models or distributed setups.
   * It assumes you (or an upstream library) can:

     * Load a linear layer’s weights.
     * Provide its inverse Hessian (or equivalent).
   * 4Bit Forge then provides the **fast qmeta builder + GPTQ solve** for that layer.

2. **MoE-Quant-aligned architecture**
   Conceptually compatible with MoE-Quant’s split:

   * Outer layer:

     * Calibration, data/expert parallelism, checkpoint plumbing.
   * Inner engine:

     * Quantization grid + GPTQ loop.

   4Bit Forge sits in the **inner engine** slot.

3. **qmeta4-centric design**

   * Instead of storing full per-element scale tensors, we store a compact 4-byte struct per **group**:

     * Q8.8 `log2(scale)`, `uint8` zero-point, and `flags`.
   * All GPU kernels and the solver consume/produce this qmeta4 format.
   * This makes metadata cheap to store, copy, and share between solver + runtime.

4. **GPU-first, CPU-parity**

   * CUDA kernels implement the fast path:

     * Warp-level butterfly reductions.
     * Vectorized loads.
     * Constant memory for candidate grids during MSE search.
   * CPU reference code mirrors semantics for correctness / parity tests.

5. **Layer-local GPTQ**

   * GPTQ is **per linear layer**:

     * `weight ∈ ℝ^{C×R}` (transposed).
     * `hessian_inv ∈ ℝ^{C×C}` for that linear.
   * No assumptions about global model structure or specific transformer architecture.

---

## 2. Repo Layout (Core)

Expected minimal structure around the implemented core:

```text
forge/
  __init__.py

  gptq.py              # GPTQ class: qmeta builder + solver
  cuda/
    __init__.py        # torch.utils.cpp_extension bindings
    quant_grid.cu      # CUDA kernels for qmeta build + MSE refinement

specifications.md      # this file
project.md             # project notes / roadmap
tests/                 # (to be expanded with correctness tests)
```

Status:

* [x] `forge/gptq.py`
* [x] `forge/cuda/quant_grid.cu`
* [ ] Dedicated test suite mirroring all CUDA/CPU paths.

---

## 3. QMeta4 Format

### 3.1 Motivation

Typical GPTQ pipelines (incl. MoE-Quant) often store:

* `scale: [..., C]` or `[..., G]` as float16/float32.
* `qzero: [..., C]` or `[..., G]` as float/ints.

For big layers, this is:

* Large in memory.
* Expensive to move across device boundaries.
* Awkward to reuse between solver + runtime (different layouts / lifecycles).

4Bit Forge collapses groupwise metadata into a **4-byte struct** per group, making it:

* Smaller to store and move (huge reduction in bandwidth).
* Naturally shared across:

  * GPTQ solver,
  * W4 matmul kernels,
  * On-disk representation.

Trade-offs:

* **Pros:**

  * ~128× less metadata per element (for group size 128).
  * Fixed-width, GPU-friendly struct.
  * Shared binary layout C++ ↔ PyTorch.
* **Cons:**

  * Q8.8 fixed precision for `log2(scale)`.
  * `qzero` limited to `uint8` (0..255).

This is fine for 4–8 bit quant with typical group sizes (G=128).

### 3.2 Binary Layout (C++)

```cpp
struct QMetaPacked {
    int16_t  log2_scale_fp;  // log2(scale) in Q8.8 fixed-point
    uint8_t  qzero;          // zero-point (0..255)
    uint8_t  flags;          // bitfield; bit0 = symmetric? others reserved
};
```

* `log2_scale_fp`

  * Represents `log2(scale)` multiplied by 256 and rounded.
  * Stored as signed int16 → ~±128 in log2 units.

* `qzero`

  * `uint8` zero-point.
  * For symmetric quant, typically `(maxq + 1)/2`.

* `flags`

  * Bit 0: `1` if symmetric quantization was requested.
  * Other bits reserved (e.g. per-group overrides, “disabled group”, etc.).

In CUDA, `qmeta_bytes` is a `torch::Tensor` of shape `[G_total, 4]` with dtype `uint8`, interpreted as an array of `QMetaPacked`.

### 3.3 Python View & Encode/Decode

Python exposes qmeta as:

* Flat: `qmeta_flat: (G_total, 4) uint8`.
* Reshaped: `qmeta: (C, num_groups, 4) uint8`.

Encoding (CPU reference):

```python
eps = 1e-12
s = torch.clamp(scale_g, min=eps)               # (G,)
log2_fp = torch.log2(s)                         # float32
log2_q88 = torch.round(log2_fp * 256.0).to(torch.int16)

lo = (log2_q88 & 0xFF).to(torch.uint8)
hi = ((log2_q88 >> 8) & 0xFF).to(torch.uint8)

qzero_u8 = qzero_g.round().clamp(0, 255).to(torch.uint8)

qmeta = torch.empty(G, 4, dtype=torch.uint8, device=device)
qmeta[:, 0] = lo
qmeta[:, 1] = hi
qmeta[:, 2] = qzero_u8
qmeta[:, 3] = 0  # flags (set by CUDA path if needed)
```

Decoding (used in solver):

```python
lo = qmeta_bytes[:, 0].to(torch.int16)
hi = qmeta_bytes[:, 1].to(torch.int16)
log2_q88 = lo | (hi << 8)
log2_fp = log2_q88.to(torch.float32) / 256.0
scale = torch.exp2(log2_fp)                    # (G,)

qzero = qmeta_bytes[:, 2].to(torch.float32)    # (G,)
```

CPU + CUDA both follow the same semantics.

### 3.4 Shape Conventions

Let:

* `C = out_features` (rows of weight).
* `R = in_features` (cols of weight).
* `group_size` (typically a multiple of 32).
* `num_groups = ceil(R / group_size)`.

Then:

* Weight into grid builder: `weight: (C, R)` → reshaped to:

  ```text
  W_pad     : (C, padded_R)
  W_groups  : (C, num_groups, group_size)
  x_groups  : (C * num_groups, group_size)  # flattened for CUDA
  ```

* qmeta returned as:

  ```text
  qmeta_flat: (C * num_groups, 4)    # uint8
  qmeta     : (C, num_groups, 4)
  ```

---

## 4. CUDA Quant Grid Kernels (`quant_grid.cu`)

### 4.1 Common Utilities

Inspired by PackBoost-style kernels.

#### Warp Reductions

```cpp
template <typename T>
__device__ __forceinline__ T butterflyReduceMin(T v) { ... }

template <typename T>
__device__ __forceinline__ T butterflyReduceMax(T v) { ... }
```

#### Type → float conversion

```cpp
template <typename T>
__device__ __forceinline__ float val_to_float(T val) {
    return static_cast<float>(val);
}

// FP8 specialization
template <>
__device__ __forceinline__ float val_to_float<uint8_t>(uint8_t val) {
    __nv_fp8_e4m3 fp8_val = *reinterpret_cast<__nv_fp8_e4m3*>(&val);
    return float(fp8_val);
}
```

#### Q8.8 helpers

```cpp
__device__ __forceinline__ int16_t encode_scale_q88(float s) {
    float log2s = fast_log2(fmaxf(s, 1e-20f));
    float fp    = log2s * 256.0f;
    fp = fminf(fmaxf(fp, -32768.0f), 32767.0f);
    return static_cast<int16_t>(lrintf(fp));
}

__device__ __forceinline__ float decode_scale_q88(int16_t q) {
    float fp = static_cast<float>(q) * (1.0f / 256.0f);
    return fast_exp2(fp);
}
```

#### Candidate grid in constant memory

```cpp
__constant__ float c_p[1024];  // up to 1024 shrink factors
```

### 4.2 Range-Based Meta Builder

Host wrapper:

```cpp
std::tuple<torch::Tensor, torch::Tensor> build_group_meta_packed_cuda(
    torch::Tensor x_groups,  // [G_total, group_size]
    int64_t bit_width,
    bool symmetric
);
```

Constraints:

* `x_groups`:

  * 2D CUDA tensor, `[G_total, group_size]`.
  * `group_size % 32 == 0`.
  * Dtype: `float32`, `float16`, `bfloat16`, or FP8 (E4M3) via `uint8_t`.

Kernel: `build_group_meta_optimized<scalar_t>`:

* One block per group.
* Steps per group:

  1. Compute `(xmin, xmax)` with vectorized loads.
  2. Compute base `scale` + `qzero` (symmetric or asymmetric).
  3. Encode into `QMetaPacked` with `encode_scale_q88`.

Outputs:

* `qmeta_tensor: [G_total, 4] uint8`.
* `maxq: scalar` = `(1 << bit_width) - 1`.

### 4.3 MSE / Lᵖ Scale Refinement

Host wrapper:

```cpp
torch::Tensor mse_scale_groups_packed_cuda(
    torch::Tensor x_groups,    // [G_total, group_size]
    torch::Tensor p,           // [P] float32, shrink factors
    torch::Tensor qmeta_bytes, // [G_total, 4] uint8
    double maxq,
    double norm
);
```

Constraints:

* `x_groups` same as above.
* `p` length ≤ 1024 → loaded into `__constant__ c_p`.

Kernel: `mse_search_kernel_nosmem<scalar_t, IS_L2_NORM>`:

* Launch:

  * One warp (32 threads) per group.
  * No shared memory (faster than SMEM variant in benchmarks).
* Per group:

  1. Decode base `scale` from qmeta.
  2. For each candidate shrink factor `p_k`:

     * Compute `s_k = base_s * p_k`.
     * Quantize + dequantize the group.
     * Accumulate Lᵖ loss (L² or general Lᵖ via log/exp).
  3. Pick best candidate; write updated `log2_scale_fp` into qmeta.

Python picks `IS_L2_NORM` when `norm == 2.0`.

---

## 5. Python GPTQ Core (`forge/gptq.py`)

### 5.1 `GPTQ.build_quant_grid(...)`

```python
@torch.no_grad()
def build_quant_grid(
    self,
    weight: torch.Tensor,      # (C, R), transposed weight
    group_size: int,
    bits: int,
    symmetric: bool = False,
    mode: str = "absmax",      # "absmax" or "mse"
    quant_max_shrink: float = 0.2,
    quant_n_grid: int = 100,
    quant_norm: float = 2.4,
) -> tuple[torch.Tensor, torch.Tensor, int]:
```

Behaviour:

1. Validate shapes & params (`ndim == 2`, `group_size % 32 == 0`, etc).
2. Ensure contiguity:

   * CUDA: keep dtype, `.contiguous()`.
   * CPU: cast to float32, `.contiguous()`.
3. Compute padding & groups.
4. Reshape to `[G_total, group_size]`.
5. Dispatch:

   * CUDA path → `build_group_meta_packed_cuda` (+ `mse_scale_groups_packed_cuda` for `mode == "mse"`).
   * CPU path → `_find_quantization_meta_groups`, `_mse_scale_groups`, `_encode_qmeta_groups`.
6. Reshape qmeta to `(C, num_groups, 4)` and return `(qmeta, maxq, pad)`.

### 5.2 `_build_quant_grid_gpu(...)`

Internal helper that:

* Asserts `x_groups.is_cuda`.
* Calls CUDA kernels.
* Builds candidate shrinkage grid `p` for MSE mode on GPU.

### 5.3 `_build_quant_grid_cpu(...)`

Internal helper that:

* Computes base scales / zeros with range-based stats.
* Optionally refines via naive CPU MSE search.
* Packs into qmeta4.

### 5.4 `GPTQ.solver(...)` (Current Implementation)

```python
@torch.no_grad()
def solver(
    self,
    weight: torch.Tensor,       # (C, R), transposed weight
    hessian_inv: torch.Tensor,  # (C, C)
    qmeta: torch.Tensor,        # (C, G, 4) uint8
    maxq: torch.Tensor,         # scalar tensor
    group_size: int,
    bits: int,
) -> torch.Tensor:
    """
    GPTQ solver (groupwise, qmeta4-based reference).
    Returns:
      qweight: (C, R) uint8 quantized codes (no bit-packing).
    """
```

Algorithm (row-wise GPTQ):

1. Setup working copies:

   * `W = weight.to(torch.float32).contiguous()`.
   * `Hinv = hessian_inv.to(torch.float32).contiguous()`.
   * `qweight = torch.empty_like(weight, dtype=torch.uint8)`.

2. Compute `num_groups` and `padded_R` from `qmeta` and `group_size`.

3. Loop over rows `j`:

   * Take `h_tail = Hinv[j, j+1:]` if `j+1 < C`, else None.

   * Decode all group meta for row `j` once:

     ```python
     qmeta_row = qmeta[j]  # (G, 4)
     scale_row, qzero_row = self._decode_qmeta_groups(qmeta_row)
     ```

   * For each group `g`:

     * Compute `[start, end)` indices.

     * Apply per-group quantization:

       ```python
       s  = scale_row[g]
       q0 = qzero_row[g]

       x = W[j, start:end]
       q = torch.round(x / s + q0)
       q.clamp_(0.0, maxq_val)
       y = (q - q0) * s
       e = y - x

       W[j, start:end] = y
       weight[j, start:end] = y.to(weight.dtype)
       qweight[j, start:end] = q.to(torch.uint8)
       ```

     * Error propagation:

       ```python
       if h_tail is not None:
           W[j+1:, start:end] += h_tail.unsqueeze(1) * e.unsqueeze(0)
       ```

4. Return `qweight`.

This is a **reference implementation**: matches GPTQ logic, but not yet fused into a custom CUDA kernel.

---

## 6. Mapping to MoE-Quant (`gptq.py`, `quant_utils.py`, `gptq_loop.py`)

This section is “how current 4Bit Forge maps onto the MoE-Quant design”.

### 6.1 MoE-Quant Pipeline (Recap)

For a given layer:

1. **Hessian accumulation**

   * `GPTQ.update(input)` builds `H ≈ 2/N Σ xᵀx` from activations.

2. **Pre-step**

   * `quantization_pre_step()`:

     * Regularizes H (damping etc).
     * Handles distributed reduction.
     * Inverts H via `linalg_utils.inv_sym` + Cholesky.
     * Handles permutations based on `QuantizationOrder`.

3. **Quant grid**

   * `quant_utils.get_quantization_grid(...)`:

     * Reshapes weight into `[..., G, group_size]`.
     * Calls `find_quantization_meta(...)` for base scales/zeros.
     * Optionally refines scales via `mse_scale(...)` (Triton).
     * Broadcasts back to full `scale, qzero` grids `[C, R]`.

4. **GPTQ solver**

   * `gptq_loop.gptq_loop(...)`:

     * Uses `quantize_error_triton` and `addvv_triton`.
     * Iterates columns in blocks, quantizes, and spreads error using `H⁻¹`.

### 6.2 4Bit Forge Mapping

Right now 4Bit Forge takes **(W, H⁻¹)** as inputs and replaces steps (3) and (4) with a qmeta4-based implementation.

| MoE-Quant Component                     | 4Bit Forge Component                                         | Status |
| --------------------------------------- | ------------------------------------------------------------ | ------ |
| `GPTQ.update(input)`                    | **External**: user / upstream handles H accumulation         | [ ]    |
| `quantization_pre_step()`               | **External**: user / upstream handles H regularize + inverse | [ ]    |
| `quant_utils.get_quantization_grid()`   | `GPTQ.build_quant_grid(...)` → `(qmeta4, maxq, pad)`         | [x]    |
| `gptq_loop.gptq_loop(...)`              | `GPTQ.solver(...)` (PyTorch reference)                       | [x]    |
| Triton `mse_scale(...)`                 | CUDA `mse_scale_groups_packed_cuda(...)` on qmeta4           | [x]    |
| Triton `quantize_error_triton`, `addvv` | Planned `qmeta4` GPTQ CUDA kernel                            | [ ]    |

So 4Bit Forge is **intentionally lower-level**:

* Expects:

  * Inverted Hessian (or factor).
  * Chosen `group_size`, `bits`, `symmetric`, `mode`.
* Provides:

  * Fast **qmeta4 grid** (ABSmax + optional MSE search).
  * GPTQ solver that consumes qmeta4.

### 6.3 Target “Finished” GPTQ Interface

Once the core is stable, the top-level GPTQ interface we’re aiming for looks roughly like:

```python
class GPTQ:
    def __init__(
        self,
        group_size: int = 128,
        bits: int = 4,
        symmetric: bool = False,
        quantization_scale: str = "absmax",  # or "mse"
    ):
        ...

    @torch.no_grad()
    def build_quant_grid(self, weight: torch.Tensor):
        """
        (C, R) -> qmeta_bytes (C, G, 4), maxq, pad
        """

    @torch.no_grad()
    def solver(
        self,
        weight_t: torch.Tensor,       # (C, R), transposed
        hessian_inv: torch.Tensor,    # (C, C)
        qmeta_bytes: torch.Tensor,    # (C, G, 4)
        maxq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns qweight (C, R) uint8.
        """

    @torch.no_grad()
    def quantize_from_hinv(
        self,
        weight: torch.Tensor,         # (d_out, d_in)
        hessian_inv: torch.Tensor,    # (d_in, d_in)
    ):
        """
        Convenience wrapper:
        - builds qmeta4
        - runs solver
        - returns (qweight, qmeta4, pad)
        """
```

On top of this, a **MoE-Quant-compatible wrapper** can implement:

* `update(input)`.
* `quantization_pre_step()`.
* `quantize(bits)`.

Without rewriting any of the core qmeta4/kernels.

---

## 7. Why qmeta4 Can Enable a Faster Solver

Right now the solver is still PyTorch-level, so most of qmeta4’s benefits are **structural**. But it’s specifically designed to make a future CUDA solver unusually strong.

### 7.1 Bandwidth & Cache Benefits

For group size `G = 128`, bits = 4:

* MoE-Quant style (full grids):

  * `scale` + `qzero` stored per element → O(C·R) metadata.
* 4Bit Forge qmeta4:

  * 4 bytes per group → O(C·(R/G)) metadata.

Per element:

* ~4 bytes (e.g. FP16 scale, FP16 qzero) → ~0.03125 bytes for qmeta4.
  That’s a **128× reduction** in metadata storage and memory traffic.

With a fused CUDA solver:

* One `QMetaPacked` load per group.
* Decode to registers once.
* Use those registers for all 128 elements in the group.

Given GPTQ is heavily **memory-bound** (H⁻¹ + W already *hurt* bandwidth), making meta effectively free gets us closer to the compute limit.

### 7.2 Cleaner CUDA Graph State

MoE-Quant’s CUDA graph captures big `scale` / `qzero` tensors.

With qmeta4:

* Graph only needs to track a relatively tiny `(C, G, 4)` tensor.
* Per-group expansion lives inside the kernel, not in the graph state.

This simplifies:

* CUDA graphs for GPTQ.
* Sharing the same metadata between:

  * GPTQ solve,
  * Int4 matmuls,
  * On-disk formats.

### 7.3 Unified Metadata Across Solver + Runtime

Instead of:

* “Solver layout” for scales.
* “Runtime layout” for scales.
* “Checkpoint layout” for scales.

We can aim for:

* One canonical **qmeta4** representation reused everywhere:

  * Solver.
  * Runtime GEMMs.
  * Saved checkpoints.

That keeps everything consistent, and all the heavy lifting is in qmeta4-aware kernels rather than juggling formats.

---

## 8. Integration Pattern (MoE-Quant-Style)

Even though 4Bit Forge doesn’t yet provide the outer calibration pipeline, it’s designed to drop into that pattern.

### 8.1 Dense Linear Layer

For a dense `nn.Linear` or equivalent:

1. Transpose:

   ```python
   W = layer.weight.data        # (d_out, d_in)
   W_t = W.transpose(0, 1)      # (C = d_in, R = d_out)
   ```

2. Obtain H⁻¹:

   ```python
   Hinv = build_or_load_hessian_inverse(layer, ...)  # (C, C)
   ```

3. Build qmeta4:

   ```python
   gptq = GPTQ(group_size=128, bits=4, symmetric=True)

   qmeta, maxq, pad = gptq.build_quant_grid(
       weight=W_t,
       group_size=128,
       bits=4,
       symmetric=True,
       mode="mse",           # or "absmax"
       quant_max_shrink=0.3,
       quant_n_grid=100,
       quant_norm=2.4,
   )
   ```

4. Solve GPTQ:

   ```python
   qweight_t = gptq.solver(
       weight=W_t,
       hessian_inv=Hinv,
       qmeta=qmeta,
       maxq=maxq,
       group_size=128,
       bits=4,
   )  # (C, R)
   ```

5. Transpose back / store:

   ```python
   qweight = qweight_t.transpose(0, 1).contiguous()  # (d_out, d_in)
   # Save qweight + qmeta + pad as your quantized representation
   ```

### 8.2 MoE Experts

For MoE MLP experts:

* You just repeat the above **per expert**, with expert-specific H⁻¹:

  ```python
  for e in range(num_experts):
      W_e_t = W_e[e].transpose(0, 1)
      Hinv_e = Hinv_list[e]

      qmeta_e, maxq_e, pad_e = gptq.build_quant_grid(...)
      qweight_e_t = gptq.solver(...)
  ```

Expert-parallelism (which GPU owns what, how routing behaves) stays outside 4Bit Forge.

---

## 9. Implementation Checklist & Roadmap

### 9.1 Implemented Components (v2.4)

* [x] **qmeta4 format**

  * C++ `QMetaPacked` struct.
  * Python/Torch encode/decode helpers.
* [x] **CUDA range meta builder**

  * `build_group_meta_optimized<scalar_t>` kernel.
  * `build_group_meta_packed_cuda(...)` host wrapper.
* [x] **CUDA MSE / Lᵖ refinement**

  * `mse_search_kernel_nosmem<scalar_t, IS_L2_NORM>`.
  * `mse_scale_groups_packed_cuda(...)` host wrapper.
* [x] **CPU reference meta logic**

  * `_find_quantization_meta_groups(...)`.
  * `_mse_scale_groups(...)`.
  * `_encode_qmeta_groups(...)`.
* [x] **Python GPTQ API**

  * `GPTQ.build_quant_grid(...)`.
  * `_build_quant_grid_gpu(...)` / `_build_quant_grid_cpu(...)`.
  * `GPTQ.solver(...)` (row-wise GPTQ using qmeta4 + Hinv).
* [x] **Multi-dtype support for grid builder**

  * FP32, FP16, BF16, FP8-E4M3 via `val_to_float`.

### 9.2 Missing Kernels / Next Steps

**1. QMeta4 GPTQ CUDA Solver**

* [ ] Design & implement a fused CUDA solver kernel that:

  * Inputs:

    * `weight (C, R)` fp32/fp16.
    * `hessian_inv (C, C)` (or Cholesky factor).
    * `qmeta_bytes (C, G, 4)`.
    * `maxq`, `group_size`, `bits`.
  * Behaviour:

    * Decodes qmeta per group into registers.
    * Runs GPTQ column/block loop entirely on device.
    * Writes `qweight (C, R)` as uint8.
  * Then wire into `GPTQ.solver(..., use_cuda_solver=True)`.

**2. MoE-Quant-Compatible Wrapper**

* [ ] Add a thin adapter:

  ```python
  class GPTQMoECompat:
      def __init__(self, layer: nn.Module, ...):
          # wraps forge.GPTQ but exposes:
          # update(input), quantization_pre_step(), quantize(bits)
  ```

* [ ] Reuse 4Bit Forge core internally for:

  * `get_quantization_grid`.
  * GPTQ loop.

**3. INT4 Packing + Runtime Bridges**

* [ ] Implement kernels/utilities to:

  * Pack `qweight (uint8)` into:

    * Generic Nibble-packed int4 format.
    * Specific layouts expected by runtime engines (e.g. Marlin, vLLM GPTQ).
* [ ] Glue code for:

  * Passing qmeta4 + packed int4 to inference kernels.

**4. Hessian / Calibration Helpers**

* [ ] Optional CUDA helpers to build H:

  * `H += XᵀX` from activation batches.
* [ ] Support for:

  * Per-expert Hessians.
  * Blockwise Hessians (for memory efficiency).

**5. Tests + Parity**

* [ ] Compare against MoE-Quant’s:

  * `get_quantization_grid`.
  * `gptq_loop`.
  * Relative MSE metrics.

---

## 10. Future Work (Streaming, Distributed, Runtime)

These are intentionally **future layers** on top of the core:

* **Streaming Calibration & Quantization**

  * Layer-wise / shard-wise streaming for 1 TB checkpoints.
  * Maintain H / H⁻¹ within VRAM+RAM budgets.
  * Use same qmeta4 core for all layers/experts.

* **Runtime Integration**

  * vLLM / custom engines:

    * Load qmeta4 + packed int4.
    * Use groupwise W4A16 kernels that decode qmeta4 on the fly.

* **Distributed / MoE Orchestration**

  * Expert-parallel calibration (per-expert H).
  * Data-parallel and tensor-parallel friendly APIs.

All of that can evolve outside this repo, as long as they respect the core contracts:

* `build_quant_grid`: (W) → (qmeta4, maxq, pad).
* `solver`: (W, H⁻¹, qmeta4) → qweight.

---

## 11. Non-Goals (For This Repo)

To keep the core sharp, this repo explicitly does **not** attempt to:

* Implement full Hugging Face / vLLM loaders.
* Encode any specific transformer/MoE architecture assumptions.
* Hard-lock a particular runtime format (that belongs in a separate “runtime” repo).
* Hide or “black-box” the Hessian; callers remain responsible for:

  * How `H` is accumulated.
  * How `H` is regularized and inverted.
  * Whether they use full, blockwise, or approximate H.

4Bit Forge is meant to be the **small, sharp GPTQ core** you slot under whatever orchestration stack you want.
