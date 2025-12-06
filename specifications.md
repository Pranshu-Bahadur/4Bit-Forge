# 4BIT FORGE: GPTQ Core & QMeta4 Specification

**Version:** 2.3  
**Focus:** qmeta4 quant grid + GPTQ solver core (MoE-Quant–style, streaming-ready design)  
**Target models:** Large transformer / MoE LLMs (e.g. DeepSeek-Math-V2) with groupwise W4A16-style weight quantization.

This document describes the current design and implementation status of 4Bit Forge’s **core GPTQ engine**, centered around:

- A compact **qmeta4** format for groupwise quantization metadata.
- Fast CUDA kernels to build / refine qmeta from weights.
- A reference **GPTQ solver** that consumes qmeta + Hessian inverse and emits quantized weights.

Higher-level pieces (checkpoint I/O, calibration / Hessian builders, MoE orchestration, runtime kernels) are explicitly in scope **for the project**, but not yet implemented in this repo.

---

## 0. High-Level Overview

### 0.1 Goals

4Bit Forge aims to be a **minimal, composable GPTQ engine** that can be plugged into a larger quantization stack (e.g. MoE-Quant-style pipeline):

- Efficient **groupwise metadata** (scale, zero-point, flags) encoded as 4 bytes per group (**qmeta4**).
- **CUDA-accelerated grid search** to choose good per-group scales (absmax or MSE/Lᵖ).
- A **row-wise GPTQ solver** that:
  - Takes a single linear weight matrix and its inverse Hessian.
  - Quantizes weights groupwise using qmeta4.
  - Propagates quantization error using the Hessian inverse (standard GPTQ logic).

Everything else (what model this comes from, where Hessians come from, how you pack INT4 for matmuls) is handled by outer tooling.

### 0.2 Scope & Implementation Status

Core functionality:

- [x] **qmeta4 binary format** (C++ + Torch-side encode/decode).
- [x] **Range-based group meta builder** (CUDA + CPU reference).
- [x] **MSE / Lᵖ grid-search refinement** (CUDA + CPU reference).
- [x] **Python GPTQ API**:
  - [x] `GPTQ.build_quant_grid(...)` → qmeta4 from raw weights.
  - [x] `GPTQ.solver(...)` → GPTQ quantization given `H⁻¹` + qmeta4.
- [x] Support for fp32 / fp16 / bf16 / fp8 (E4M3) input weights in CUDA path.
- [ ] Hessian / calibration utilities.
- [ ] MoE-specific helpers (expert routing, per-expert Hessians).
- [ ] Checkpoint streaming I/O helpers (safetensors shards, etc.).
- [ ] INT4 packing and W4A16 matmul runtime kernels.
- [ ] True single-GPU streaming quantization (layer-wise weight/activation streaming).

Design-wise, the core is **“streaming-ready”** and MoE-compatible: we enforce shapes / contracts that can be fed by a streaming calibration stack later, but v2.3 itself is an **offline GPTQ kernel core**.

---

## 1. Design Principles

1. **Core, not framework (for now)**  
   - This repo does *not* load full checkpoints, run full-model calibration, or implement distributed MoE orchestration.
   - It **assumes** you (or an upstream library) can:
     - Load a linear layer’s weights.
     - Provide its inverse Hessian (or equivalent).
   - 4Bit Forge then provides the **fast qmeta builder + GPTQ solve** for that layer.

2. **MoE-Quant-aligned architecture**  
   - Conceptually similar to MoE-Quant’s separation of concerns:
     - High-level scripts handle calibration, data/expert parallelism, and checkpoint plumbing.
     - Low-level kernels handle GPTQ’s heavy lifting.
   - 4Bit Forge is meant to drop into that “low-level GPTQ engine” slot.

3. **qmeta4-centric design**  
   - Instead of storing full per-group scale tensors (`float32`/`float16`), we store a compact 4-byte struct:
     - Q8.8 `log2(scale)`, `uint8` zero-point, and `flags`.
   - All GPU kernels and the solver consume/produce this qmeta4 format.
   - This makes metadata cheap to store, copy, and ship across devices.

4. **GPU-first, CPU-parity**  
   - CUDA kernels implement the fast path, using:
     - Warp-level butterfly reductions.
     - Vectorized loads.
     - Constant memory for candidate grids.
   - CPU reference code mirrors the exact semantics for correctness / testing.

5. **Layer-local GPTQ**  
   - All GPTQ operations are **per linear layer**:
     - `weight ∈ ℝ^{C×R}` (transposed).
     - `hessian_inv ∈ ℝ^{C×C}` for that linear.
     - No assumptions about global model structure.

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
````

Status:

* [x] `forge/gptq.py`
* [x] `forge/cuda/quant_grid.cu`
* [ ] Dedicated test suite mirroring all paths.

---

## 3. QMeta4 Format

### 3.1 Motivation for qmeta4

Earlier GPTQ-style pipelines often stored:

* `scale: [C, G]` (or `[G, C]`) as float32/float16.
* `qzero: [C, G]` as float32/int32.

This is flexible but heavy in memory and I/O.

4Bit Forge collapses groupwise metadata into a **4-byte struct** per group, making it cheap to store and move while remaining easy to decode on GPU.

The core trade-offs:

* **Pros:**

  * ~4× smaller metadata vs `float32` scale tensors.
  * GPU-friendly fixed-size struct.
  * Same binary layout shared by CUDA & Torch.
* **Cons:**

  * Fixed Q8.8 resolution for `log2(scale)`.
  * `qzero` limited to `uint8` (0..255).

This is acceptable for 4–8 bit quantization with typical group sizes (e.g. 128).

### 3.2 Binary layout (C++)

```cpp
struct QMetaPacked {
    int16_t  log2_scale_fp;  // log2(scale) in Q8.8 fixed-point
    uint8_t  qzero;          // zero-point (0..255)
    uint8_t  flags;          // bitfield; bit0 = symmetric? others reserved
};
```

* `log2_scale_fp`:

  * Represents `log2(scale)` multiplied by 256 and rounded.
  * Stored as signed int16 to cover a ~±128 range in log2 units.
* `qzero`:

  * `uint8` zero-point.
  * For symmetric quant, typically the midpoint code `(maxq + 1)/2`.
* `flags`:

  * Bit 0: `1` if symmetric quantization was requested.
  * Remaining bits reserved for future use (per-group overrides, etc.).

In CUDA, `qmeta_bytes` is a `torch::Tensor` of shape `[G_total, 4]` with dtype `uint8`, interpreted as an array of `QMetaPacked`.

### 3.3 Python view & encode/decode

On the Python side, qmeta is surfaced as:

* Flat: `qmeta_flat: (G_total, 4) uint8`.
* Reshaped: `qmeta: (C, num_groups, 4) uint8`.

Encoding (CPU reference, `_encode_qmeta_groups`):

```python
eps = 1e-12
s = torch.clamp(scale_g, min=eps)               # (G,)
log2_fp = torch.log2(s)                         # float32
log2_q88 = torch.round(log2_fp * 256.0).to(torch.int16)

lo = (log2_q88 & 0xFF).to(torch.uint8)         # low byte
hi = ((log2_q88 >> 8) & 0xFF).to(torch.uint8)  # high byte

qzero_u8 = qzero_g.round().clamp(0, 255).to(torch.uint8)

qmeta = torch.empty(G, 4, dtype=torch.uint8, device=device)
qmeta[:, 0] = lo
qmeta[:, 1] = hi
qmeta[:, 2] = qzero_u8
qmeta[:, 3] = 0  # flags currently not set from CPU path
```

Decoding (used in solver):

```python
lo = qmeta_bytes[:, 0].to(torch.int16)
hi = qmeta_bytes[:, 1].to(torch.int16)
log2_q88 = lo | (hi << 8)
log2_fp = log2_q88.to(torch.float32) / 256.0   # back to log2(scale)
scale = torch.exp2(log2_fp)                    # (G,)

qzero = qmeta_bytes[:, 2].to(torch.float32)    # (G,)
```

This ensures **exact parity** between CPU and CUDA paths for qmeta semantics.

### 3.4 Shape conventions

* Let:

  * `C = out_features` (rows of weight).
  * `R = in_features` (columns of weight).
  * `group_size` (must be a multiple of 32).
  * `num_groups = ceil(R / group_size)`.

Then:

* Weight passed into grid builder: `weight: (C, R)` → reshaped into:

  * `x_groups: (C * num_groups, group_size)`.
* qmeta returned as:

  * `qmeta_flat: (C * num_groups, 4)` → `.view(C, num_groups, 4)`.

---

## 4. CUDA Quant Grid Kernels (`quant_grid.cu`)

### 4.1 Common utilities

Implementation draws inspiration from PackBoost-style kernels:

* Warp-level butterfly reductions:

  ```cpp
  template <typename T>
  __device__ __forceinline__ T butterflyReduceSum(T val) { ... }

  template <typename T>
  __device__ __forceinline__ T butterflyReduceMin(T val) { ... }

  template <typename T>
  __device__ __forceinline__ T butterflyReduceMax(T val) { ... }
  ```

* Type-to-float conversion:

  ```cpp
  template <typename T>
  __device__ __forceinline__ float val_to_float(T val) {
      return static_cast<float>(val);
  }

  // FP8 specialization (assuming __nv_fp8_e4m3)
  template <>
  __device__ __forceinline__ float val_to_float<uint8_t>(uint8_t val) {
      __nv_fp8_e4m3 fp8_val = *reinterpret_cast<__nv_fp8_e4m3*>(&val);
      return float(fp8_val);
  }
  ```

* Q8.8 log2 scale encoding/decoding:

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

* Candidate grid in constant memory:

  ```cpp
  __constant__ float c_p[1024];   // up to 1024 shrink factors
  ```

### 4.2 Range-based meta builder

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

  * 2D CUDA tensor.
  * Shape `[G_total, group_size]`.
* `group_size % 32 == 0`.
* Dtype can be:

  * `float32`, `float16`, `bfloat16`.
  * `float8_e4m3fn` / `float8_e4m3fnuz` (via `uint8_t` reinterpretation).

Kernel (`build_group_meta_optimized<scalar_t>`):

* One block per group `g`.
* Threads:

  * Up to 256 threads per block, aligned to warp size (32).
* Steps:

  1. Compute `(xmin, xmax)` for the group:

     * Use 16-byte (`int4`) vectorized loads when possible.
     * Convert to float with `val_to_float`.
     * Track local min/max per thread.
     * Warp-reduce to global `local_min`, `local_max`.

  2. Compute `scale` (`s`) and `qzero` (`q0`) as in §3.2:

     * Symmetric vs asymmetric path.
     * Add small `eps` to avoid zero scale.

  3. Encode into `QMetaPacked`:

     * `log2_scale_fp = encode_scale_q88(s)`.
     * `qzero = uint8(clamped(q0))`.
     * `flags = symmetric ? 1 : 0`.

Outputs:

* `qmeta_tensor: [G_total, 4]` `uint8` (view of `QMetaPacked`).
* `maxq: scalar float32` equal to `(1 << bit_width) - 1`.

### 4.3 MSE / Lᵖ scale refinement

Host wrapper:

```cpp
torch::Tensor mse_scale_groups_packed_cuda(
    torch::Tensor x_groups,    // [G_total, group_size]
    torch::Tensor p,           // [P] float32, shrink factors
    torch::Tensor qmeta_bytes, // [G_total, 4] uint8
    double maxq,
    double norm                // exponent p in L^p
);
```

Constraints:

* `x_groups`:

  * Same shape & dtype constraints as in the range builder.
* `p`:

  * 1D float32 CUDA tensor.
  * `0 < P <= 1024`.
  * Copied to `__constant__ c_p` via `cudaMemcpyToSymbol`.
* `group_size % 32 == 0`.

Kernel (`mse_search_kernel_nosmem<scalar_t, IS_L2_NORM>`):

* Launch:

  * One warp (`threads = 32`) per group (`blocks = G_total`).
  * `smem_bytes = 0` — no shared memory used.
* For each group `g`:

  1. Load base metadata `m = qmeta[g]` and decode:

     * `base_s = decode_scale_q88(m.log2_scale_fp)`.
     * `q0 = float(m.qzero)`.

  2. For each candidate `p_k`:

     * `s_k = base_s * p_k`.
     * `rcp_k = 1.0f / s_k`.
     * Each lane loops over `idx = lane; idx < group_size; idx += 32`:

       * `v = val_to_float(x[base + idx])`.

       * Quantize:

         ```cpp
         float q = fast_round(fmaf(v, rcp_k, q0));
         q       = fminf(fmaxf(q, 0.0f), maxq);
         float diff = fmaf(q - q0, s_k, -v);
         ```

       * Loss contribution:

         * If `IS_L2_NORM`:

           ```cpp
           loss += diff * diff;
           ```
         * Else:

           ```cpp
           float e = fmaxf(fabsf(diff), 1e-20f);
           float lg = __logf(e);
           float val = __expf(lg * norm);
           loss += val;
           ```
     * Warp-reduce `loss` and update `(best_loss, best_s)` on lane 0.

  3. After all candidates:

     * Lane 0 writes updated `log2_scale_fp = encode_scale_q88(best_s)` back into `qmeta[g]`.

Python side determines whether `IS_L2_NORM` is used by checking `abs(norm - 2.0) < 1e-5`.

### 4.4 Type dispatch

In the host wrappers:

* For fp8:

  ```cpp
  const uint8_t* x_ptr = reinterpret_cast<uint8_t*>(x_groups.data_ptr());
  build_group_meta_optimized<uint8_t><<<...>>>(x_ptr, ...);
  ```

* For other dtypes:

  ```cpp
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, dtype, "...",
      [&]() {
          using scalar_t_ = scalar_t;
          const scalar_t_* x_ptr = x_groups.data_ptr<scalar_t_>();
          build_group_meta_optimized<scalar_t_><<<...>>>(x_ptr, ...);
      }
  );
  ```

Same pattern is used for the MSE kernel.

---

## 5. Python GPTQ Core (`forge/gptq.py`)

### 5.1 `GPTQ.build_quant_grid(...)`

Signature:

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

Behavior:

1. Validate input:

   * `weight.ndim == 2`.
   * `group_size > 0`, `group_size <= R`.
   * `group_size % 32 == 0`.

2. Ensure dtype / contiguity:

   * CUDA: keep original dtype (fp16/bf16/fp32/fp8) and call `.contiguous()`.
   * CPU: cast to `float32` and `.contiguous()`.

3. Compute grouping:

   ```python
   C, R = weight.shape
   num_groups = (R + group_size - 1) // group_size
   padded_R   = num_groups * group_size
   pad        = padded_R - R
   ```

4. Pad last dimension if needed:

   ```python
   W_pad = F.pad(weight, (0, pad)) if pad > 0 else weight
   ```

5. Reshape into groups:

   ```python
   W_groups = W_pad.view(C, num_groups, group_size)
   x_groups = W_groups.reshape(-1, group_size)  # (G_total, group_size)
   ```

6. GPU vs CPU paths:

   * **CUDA:**

     ```python
     qmeta_bytes, maxq = self._build_quant_grid_gpu(
         x_groups, bits, symmetric, mode,
         quant_max_shrink, quant_n_grid, quant_norm,
     )
     ```

     `_build_quant_grid_gpu` calls:

     * `kernels.build_group_meta_packed(...)`.
     * Optionally `kernels.mse_scale_groups_packed(...)` if `mode == "mse"`.

   * **CPU:**

     ```python
     qmeta_bytes, maxq = self._build_quant_grid_cpu(
         x_groups, bits, symmetric, mode,
         quant_max_shrink, quant_n_grid, quant_norm,
     )
     ```

     `_build_quant_grid_cpu` calls:

     * `_find_quantization_meta_groups(...)` (range-based).
     * Optionally `_mse_scale_groups(...)` (naive nested loops).
     * `_encode_qmeta_groups(...)` to get qmeta bytes.

7. Reshape qmeta:

   ```python
   qmeta = qmeta_flat.view(C, num_groups, 4)
   ```

Return:

* `qmeta: (C, num_groups, 4) uint8`.
* `maxq: scalar float32` (`2**bits - 1`).
* `pad: int` (# of padded columns at tail, to be ignored by caller).

### 5.2 `_build_quant_grid_gpu(...)`

Key details:

* Enforces `x_groups.is_cuda` and `(G, group_size)` shape.

* Calls CUDA kernels with **no dtype upcasts** on weights (except internal float conversion via `val_to_float`).

* For MSE mode constructs candidate grid:

  ```python
  p = torch.linspace(
      1.0,
      quant_max_shrink,
      quant_n_grid,
      dtype=torch.float32,   # necessary for cudaMemcpyToSymbol
      device=device,
  )
  ```

* MSE search is weight-only, unweighted Lᵖ error (no Hessian weighting).

### 5.3 `_build_quant_grid_cpu(...)`

* Uses `_find_quantization_meta_groups` to compute initial `scale_g`, `qzero_g`.
* Optional `_mse_scale_groups` to refine `scale_g` using CPU nested loops.
* Packs `(scale_g, qzero_g)` into qmeta4 via `_encode_qmeta_groups`.

### 5.4 `GPTQ.solver(...)`

Signature:

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

Constraints:

* `weight.ndim == 2`.
* `hessian_inv.shape == (C, C)`.
* `qmeta.ndim == 3` and `qmeta.size(0) == C`, `qmeta.size(2) == 4`.
* `group_size % 32 == 0`.

Algorithm (row-wise GPTQ):

1. Shapes and copies:

   * `C, R = weight.shape`.
   * `W = weight.to(torch.float32).contiguous()` (working copy).
   * `Hinv = hessian_inv.to(torch.float32).contiguous()`.
   * `qweight = empty_like(weight, dtype=torch.uint8)`.

2. Compute `num_groups` and padding:

   ```python
   num_groups = qmeta.size(1)
   padded_R   = num_groups * group_size
   pad        = padded_R - R
   # pad should be >= 0; solver ignores padded tail
   ```

3. For each row `j` in `[0, C)`:

   * If `j + 1 < C`: `h_tail = Hinv[j, j+1:]` else `h_tail = None`.

   * Decode all group meta for this row once:

     ```python
     qmeta_row = qmeta[j]                     # (G, 4)
     scale_row, qzero_row = self._decode_qmeta_groups(qmeta_row)
     ```

   * For each group `g` in `[0, num_groups)`:

     ```python
     start = g * group_size
     end   = min(start + group_size, R)
     if start >= R or end <= start:
         continue
     ```

     * Scalar meta:

       ```python
       s  = scale_row[g]
       q0 = qzero_row[g]
       ```

     * Slice:

       ```python
       x = W[j, start:end]                      # fp32
       q = torch.round(x / s + q0)
       q.clamp_(0.0, maxq_val)
       y = (q - q0) * s
       e = y - x
       ```

     * Write-back:

       ```python
       W[j, start:end] = y
       weight[j, start:end] = y.to(w_dtype)
       qweight[j, start:end] = q.to(torch.uint8)
       ```

     * Error propagation (standard GPTQ):

       ```python
       if h_tail is not None:
           W[j+1:, start:end] += h_tail.unsqueeze(1) * e.unsqueeze(0)
       ```

4. Return `qweight`.

Notes:

* This is a **reference implementation**: correct and readable, not aggressively optimized.
* It operates per-row and per-group, fully leveraging qmeta4 for meta decoding.
* Currently uses a **global (C×C) inverse Hessian**; block-wise Hessian formats can be integrated later.

---

## 6. Integration Pattern (MoE-Quant-style)

Even though 4Bit Forge doesn’t yet provide high-level orchestration, it’s designed to slot into a MoE-Quant-like stack.

### 6.1 Dense linear layer

For each dense linear weight `W_orig ∈ ℝ^{out×in}`:

1. Transpose to `W = W_orig.T` if needed, so `W.shape == (C, R)`.

2. Compute / load `Hinv ∈ ℝ^{C×C}` from your calibration logic.

3. Build qmeta4:

   ```python
   gptq = GPTQ()

   qmeta, maxq, pad = gptq.build_quant_grid(
       weight=W,             # (C, R)
       group_size=128,
       bits=4,
       symmetric=True,
       mode="mse",           # or "absmax"
       quant_max_shrink=0.3,
       quant_n_grid=100,
       quant_norm=2.4,
   )
   ```

4. Run GPTQ:

   ```python
   qweight = gptq.solver(
       weight=W,
       hessian_inv=Hinv,
       qmeta=qmeta,
       maxq=maxq,
       group_size=128,
       bits=4,
   )
   ```

5. Store:

   * Dequantized `weight` (if you want an FP32/FP16 approximation).
   * `qweight` (uint8 codes, low bits used).
   * `qmeta` (qmeta4).
   * Or optionally transform to your own runtime format.

### 6.2 MoE experts

For MoE layers, you typically have:

* `E` experts, each with its own MLP weights (`W_e`).
* Expert-specific Hessians `Hinv_e` based on routed activations.

Integration pattern:

* For each expert `e` and its relevant linear `W_e`:

  1. Build qmeta:

     ```python
     qmeta_e, maxq_e, pad_e = gptq.build_quant_grid(
         weight=W_e,
         group_size=128,
         bits=4,
         symmetric=True,
         mode="mse",
         ...
     )
     ```

  2. Run solver:

     ```python
     qweight_e = gptq.solver(
         weight=W_e,
         hessian_inv=Hinv_e,
         qmeta=qmeta_e,
         maxq=maxq_e,
         group_size=128,
         bits=4,
     )
     ```

* Expert-parallelism (which GPU owns which experts, how calibration batches are routed) remains in the **outer stack**, not in 4Bit Forge.

---

## 7. Implementation Checklist & Roadmap

### 7.1 Implemented components (v2.3)

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

### 7.2 Planned / future components

* [ ] **Hessian / calibration utilities**

  * Streaming `XᵀX` builders from activation iterators.
  * Blockwise Hessian formats for efficiency.

* [ ] **MoE-specific helpers**

  * Utilities for expert-wise activation bucketing.
  * Per-expert calibration / Hessian management API.

* [ ] **Checkpoint I/O helpers**

  * Iterators over safetensors / bin shards.
  * Sharded save of `qweight + qmeta`.

* [ ] **Runtime path**

  * INT4 packing from `qweight` + qmeta4.
  * W4A16 group-gemm kernels for inference.

* [ ] **Streaming / distributed orchestration**

  * Single-GPU streaming quantization (layer-wise weights + activations).
  * Multi-GPU expert/data-parallel APIs modelled after MoE-Quant.

---

## 8. Non-Goals (for this repo)

To keep the core small and composable, 4Bit Forge **does not** (and will not, in this repo):

* Implement a full HF/vLLM model loader or training loop.
* Encode any particular MoE architecture assumptions into the core.
* Hard-code runtime formats for any specific serving engine.
* Hide the Hessian: the caller remains responsible for how `H⁻¹` is computed, regularized, and stored.

Those responsibilities live in **outer layers / separate libraries** that can evolve independently while relying on this core GPTQ engine as a stable building block.

```