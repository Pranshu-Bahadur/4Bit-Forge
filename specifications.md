Here is the updated document. I have converted the mathematical notation to strictly use `$` for inline math and `$$` for block math, which ensures proper rendering on GitHub and most Markdown previews. I also fixed a few syntax errors (like stray asterisks inside equations) that were breaking the logic.

-----

# 4BIT FORGE: GPTQ Core & QMeta4 Specification

**Version:** 2.5
**Focus:** qmeta4 quant grid + GPTQ solver core (MoE-Quant–style, streaming-ready design)
**Target models:** Large transformer / MoE LLMs (e.g. DeepSeek-Math-V2) with groupwise W4A16-style weight quantization.

This document describes the current design and implementation status of 4Bit Forge’s **core GPTQ engine**, centered around:

  * A compact **qmeta4** format for groupwise quantization metadata.
  * Fast CUDA kernels to build / refine qmeta from weights.
  * A GPTQ **solver** that consumes qmeta + Hessian inverse and emits quantized weights, while staying mathematically faithful to the original GPTQ algorithm.

Higher-level pieces (checkpoint I/O, calibration / Hessian builders, MoE orchestration, runtime kernels) are explicitly in scope for the *project*, but not implemented in this repo yet.

-----

## 0\. High-Level Overview

### 0.1 Goals

4Bit Forge aims to be a **minimal, composable GPTQ engine** that can be plugged into a larger quantization stack (MoE-Quant, custom GPTQ tools, vLLM loaders, etc.):

  * Efficient **groupwise metadata** (scale, zero-point, flags) encoded as 4 bytes per group (**qmeta4**).

  * **CUDA-accelerated grid search** to choose per-group scales (ABSMAX or MSE / $L^p$).

  * A **row-wise GPTQ solver** that:

      * Takes a single linear weight matrix and its inverse Hessian (or factor).
      * Quantizes weights *groupwise* using qmeta4.
      * Propagates quantization error using the Hessian inverse (same logic as GPTQ).

Everything else (which model this layer belongs to, where Hessians come from, how you pack INT4 for matmuls) is handled by outer tooling.

### 0.2 Scope & Implementation Status

Core functionality:

  * ✅ **qmeta4 binary format** (C++ + Torch-side encode/decode).

  * ✅ **Range-based group meta builder** (CUDA + CPU reference).

  * ✅ **MSE / $L^p$ grid-search refinement** (CUDA + CPU reference).

  * ✅ **Python GPTQ API:**

      * `GPTQ.build_quant_grid(...)` → qmeta4 from raw weights.
      * `GPTQ.solver(...)` → GPTQ quantization given $H^{-1}$ + qmeta4 (PyTorch reference).

  * ✅ Support for fp32 / fp16 / bf16 / fp8(E4M3) input weights in CUDA path.

  * ⬜ Hessian / calibration utilities (`update(input)`, `quantization_pre_step()`).

  * ⬜ MoE-specific helpers (expert routing, per-expert Hessians).

  * ⬜ Checkpoint streaming I/O helpers (safetensors shards, etc.).

  * ⬜ INT4 packing and W4A16 matmul runtime kernels.

  * ⬜ Fused CUDA GPTQ solver kernel (Phase-2).

  * ⬜ Potential Hessian-/group-aware refinements that leverage qmeta4 and log2 Q8.8 more deeply (Phase-3).

Design-wise, the core is **“streaming-ready”** and MoE-compatible: we enforce shapes/contracts that fit a streaming calibration stack later, but v2.5 itself is an **offline GPTQ kernel core**.

### 0.3 Non-Goals (for this repo)

4Bit Forge core does **not**:

  * Load full LLM checkpoints (no HF/vLLM loader).
  * Run end-to-end calibration or accumulate Hessians from activations.
  * Implement MoE routing / expert-parallel orchestration.
  * Pack INT4 or implement matmul kernels.
  * Implement a full quantization CLI or training pipeline.

It is designed to be plugged into a larger stack.

-----

## 1\. Design Principles

1.  **Core, not framework**

    The repo assumes you (or an upstream library) can:

      * Load a linear layer’s weights.
      * Provide its inverse Hessian (or equivalent).

    4Bit Forge then provides the **fast qmeta builder + GPTQ solve** for that layer.

2.  **MoE-Quant-aligned architecture**

    Conceptually compatible with MoE-Quant’s split:

      * Outer layer: calibration, expert/data parallelism, checkpoint plumbing.
      * Inner engine: quantization grid + GPTQ loop.

    4Bit Forge sits in the **inner engine** slot.

3.  **qmeta4-centric design**

      * Instead of storing full per-element scale tensors, we store a compact 4-byte struct per **group**:

          * Q8.8 ($\log_2(\text{scale})$), `uint8` zero-point, and flags.

      * All GPU kernels and the solver consume/produce this qmeta4 format.

      * This makes metadata cheap to store, copy, and share between solver + runtime.

4.  **GPU-first, CPU-parity**

      * CUDA kernels implement the fast path:

          * Warp-level butterfly reductions.
          * Vectorized loads.
          * Constant memory for candidate grids during MSE search.

      * CPU reference code mirrors semantics for correctness / parity tests.

5.  **Layer-local GPTQ**

      * GPTQ is **per linear layer**:

          * weight $W \in \mathbb{R}^{C \times R}$ (transposed vs PyTorch).
          * inverse Hessian $H^{-1} \in \mathbb{R}^{C \times C}$ for that input dimension.

      * No assumptions about global model structure or specific transformer architecture.

-----

## 2\. Repo Layout (Core)

Minimal structure around the implemented core:

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

  * ✅ `forge/gptq.py`
  * ✅ `forge/cuda/quant_grid.cu`
  * ⬜ Dedicated test suite mirroring all CUDA/CPU paths.

-----

## 3\. Original GPTQ Algorithm (Reference)

The original GPTQ paper (Frantar et al.) gives Algorithm 1, “Quantize $W$ given inverse Hessian $H^{-1} = (2XX^\top + \lambda I)^{-1}$ and blocksize $B$.”

We restate it here in text-friendly math.

Let:

  * $W \in \mathbb{R}^{d_{\text{row}} \times d_{\text{col}}}$ be the full-precision weight matrix.
  * $H^{-1} \in \mathbb{R}^{d_{\text{col}} \times d_{\text{col}}}$ be the inverse Hessian in the *column* space.
  * $B$ be a block size (number of columns per block).
  * `quant(·)` be the scalar/vector quantizer (e.g. 4-bit uniform with per-column or per-group scales).

Algorithm (conceptual):

1.  Initialize:

      * Quantized weights $Q \gets 0_{d_{\text{row}} \times d_{\text{col}}}$.
      * Block errors $E \gets 0_{d_{\text{row}} \times B}$.

2.  Optionally factor $H^{-1}$ via Cholesky (for stability).

3.  For each block start index $i = 0, B, 2B, \ldots$:

      * For each column $j = i, \ldots, i + B - 1$:

        1.  **Quantize column $j$:**

            $$Q_{:,j} \gets \text{quant}(W_{:,j})$$

        2.  **Compute quantization error scaled by Hessian diagonal:**

            $$E_{:, j-i} \gets \frac{W_{:,j} - Q_{:,j}}{(H^{-1})_{jj}}$$

        3.  **Update the remaining columns *within the block*:**

            $$W_{:, j+1:(i+B)} \gets W_{:, j+1:(i+B)} - E_{:, j-i} \cdot H^{-1}_{j, j+1:(i+B)}$$

      * After finishing columns in the block, **update all later columns**:

        $$W_{:, (i+B):} \gets W_{:, (i+B):} - E \cdot H^{-1}_{i:(i+B), (i+B):}$$

Intuition:

  * Quantizing column $j$ produces an error vector $e_j = W_{:,j} - Q_{:,j}$.
  * GPTQ uses the inverse Hessian to push this error into the remaining columns so that the *overall* loss in the local quadratic approximation is minimized.

-----

## 4\. How MoE-Quant Implements GPTQ

MoE-Quant follows the same math but uses slightly different shapes / naming:

  * They work with **transposed weights**:

      * `weight` has shape `(C, R)` where $C = d_{\text{col}}$ (input dim) and $R = d_{\text{row}}$ (output dim).

  * The Hessian inverse `hessian_inv` has shape `(C, C)` — same dimension as the first axis of `weight`.

So in MoE-Quant:

  * Each "GPTQ coordinate" is a row index $j \in \{0, \ldots, C-1\}$.
  * Each row vector `weight[j]` plays the role of a column $W_{:,j}$ in the original paper.

### 4.1 Quantization step (`quantize_error_triton`)

For each row $j$:

  * Inputs:

      * `weight[j]` (shape `(R,)`).
      * `scale[j]`, `qzero[j]` (shape `(R,)`), already expanded from groupwise scales.
      * `maxq`.

  * Operations:

      * Quantize elementwise:

        $$q_{j,r} = \text{clip}\Big(\text{round}\big(\frac{w_{j,r}}{s_{j,r}} + z_{j,r}\big), 0, \text{maxq}\Big)$$

      * Dequantize:

        $$\hat{w}_{j,r} = (q_{j,r} - z_{j,r}) \cdot s_{j,r}$$

      * Error vector:

        $$e_j = \hat{w}_j - w_j$$

### 4.2 Error propagation (`addvv_triton` + `addmm_`)

Given `Hinv`:

  * For rows within the current block $[i_1, i_2)$:

    $$W_{r,:} \leftarrow W_{r,:} + H^{-1}_{j,r} \cdot e_j \quad\text{for } r \in (j, i_2)$$

  * For rows outside the block $r \ge i_2$, they use a batched matmul:

    $$W_{i_2:, :} \leftarrow W_{i_2:, :} - (H^{-1}_{i_1:i_2, i_2:})^\top \cdot E_{\text{block}}$$

This is the same **outer-product update** as Algorithm 1, just with:

  * transposed weights `(C, R)` instead of $(d_{\text{row}}, d_{\text{col}})$, and
  * a blockwise implementation for efficiency.

-----

## 5\. 4Bit Forge QMeta4 Format

### 5.1 Motivation

Typical GPTQ pipelines (incl. MoE-Quant) often store:

  * `scale: (C, R)` as fp16/fp32,
  * `qzero: (C, R)` as fp16/fp32/int.

For big layers, this is:

  * Large in memory,
  * Expensive to move across device boundaries,
  * Awkward to reuse between solver + runtime.

4Bit Forge collapses groupwise metadata into a **4-byte struct** per group, making it:

  * Much smaller to store and move,

  * Naturally shared across:

      * GPTQ solver,
      * W4 matmul kernels,
      * On-disk representation.

Trade-offs:

  * Pros:

      * For group size $G = 128$, metadata is $\approx 128\times$ smaller per element vs per-element scales.
      * Fixed-width, GPU-friendly struct.
      * Shared binary layout C++ ↔ PyTorch.

  * Cons:

      * Q8.8 fixed precision for $\log_2(\text{scale})$.
      * `qzero` limited to 0…255.

### 5.2 Binary Layout

```cpp
struct QMetaPacked {
    int16_t  log2_scale_fp;  // log2(scale) in Q8.8 fixed-point
    uint8_t  qzero;          // zero-point (0..255)
    uint8_t  flags;          // bitfield; bit0 = symmetric? others reserved
};
```

**Encoding:**

Let $s > 0$ be the floating-point scale.

1.  Compute $\ell = \log_2(s)$.

2.  Fixed-point value:

    $$q = \text{round}( \ell \cdot 256 )$$

3.  Store:

      * `log2_scale_fp = (int16) q`.
      * `qzero` = rounded/clamped zero-point.
      * `flags` = bitfield.

**Decoding:**

Given `log2_scale_fp = q`:

$$\ell = \frac{q}{256}, \qquad s = 2^{\ell}$$

The relative quantization error on `scale` from this encoding is on the order of $\approx 10^{-3}$, negligible compared to 4-bit quantization noise.

### 5.3 Shape Conventions

Let:

  * $C =$ number of GPTQ coordinates (input dim),
  * $R =$ number of outputs (fan-out),
  * `group_size` divides $R$,
  * `num_groups` = $\lceil R / \text{group_size} \rceil$.

Then we use reshapes:

  * Weight into grid builder:

      * `weight`: `(C, R)`.

      * Pad to `padded_R` if needed.

      * Reshape to groups:

        ```text
        W_groups  : (C, num_groups, group_size)
        x_groups  : (C * num_groups, group_size)   # flattened for CUDA
        ```

  * qmeta returned as:

    ```text
    qmeta_flat: (C * num_groups, 4)  # uint8
    qmeta     : (C, num_groups, 4)
    ```

Each $(j, g)$ pair (row $j$, group $g$) has one `QMetaPacked` entry.

-----

## 6\. CUDA Quant Grid Kernels (`quant_grid.cu`)

### 6.1 Shared Utilities

**Warp reductions:**

  * `butterflyReduceMin`, `butterflyReduceMax`, `butterflyReduceSum` implement warp-level reductions using `__shfl_down_sync`.

**Type → float conversion:**

  * Generic template `val_to_float(T)` casting to float.

  * Specialization for FP8 E4M3 stored as `uint8_t`:

      * Reinterpret as `__nv_fp8_e4m3` and convert to float.

**Q8.8 helpers:**

  * `encode_scale_q88(float s)`: encodes $\log_2(s)$ into `int16`.
  * `decode_scale_q88(int16_t q)`: decodes back.

**Candidate grid:**

  * `__constant__ float c_p[1024];` for up to 1024 candidate shrink factors used in MSE/ $L^p$ search.

### 6.2 Range-Based Meta Builder

Host wrapper:

```cpp
std::tuple<torch::Tensor, torch::Tensor> build_group_meta_packed_cuda(
    torch::Tensor x_groups,  // [G_total, group_size]
    int64_t bit_width,
    bool symmetric
);
```

Constraints:

  * `x_groups`: `[G_total, group_size]`, CUDA, contiguous.
  * `group_size % 32 == 0`.
  * Dtype: float32/float16/bfloat16 or fp8(E4M3) (via `uint8_t`).

Kernel `build_group_meta_optimized<scalar_t>`:

For each group $g$:

1.  Compute:

    $$x_{\min} = \min_k x_{g,k}, \quad x_{\max} = \max_k x_{g,k}$$

2.  Compute base `scale` and `qzero`:

      * **Symmetric:**

          * $a_{\max} = \max(|x_{\min}|, |x_{\max}|)$.
          * $s = \frac{2}{\text{maxq}} a_{\max} + \varepsilon$.
          * $q_0 = \frac{\text{maxq} + 1}{2}$.

      * **Asymmetric:**

          * $s = \frac{x_{\max} - x_{\min}}{\text{maxq}} + \varepsilon$.
          * $q = -x_{\min} / s$.
          * Clamp $q$ to $[0, \text{maxq}]$ and round to get $q_0$.

3.  Encode:

      * `log2_scale_fp = encode_scale_q88(s)`,
      * `qzero = (uint8) round(q_0)`,
      * `flags` bit 0 = `symmetric`.

Outputs:

  * `qmeta_tensor: (G_total, 4)` uint8.
  * `maxq: scalar` with value $2^{\text{bits}} - 1$.

### 6.3 MSE / $L^p$ Scale Refinement

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

  * `p` is a grid of shrink factors, e.g. `torch.linspace(1-quant_max_shrink, 1+quant_max_shrink, quant_n_grid)`.

Kernel `mse_search_kernel_nosmem<scalar_t, IS_L2_NORM>`:

For each group $g$:

1.  Decode base scale $s_{\text{base}}$ from qmeta:

    $$s_{\text{base}} = \text{decode_scale_q88}(\text{log2_scale_fp})$$

2.  For each candidate factor $p_k$:

      * Test scale $s_k = s_{\text{base}} \cdot p_k$.

      * For each element $x_{g,k}$ in the group:

          * Quantize:

            $$q = \text{round}\Big(\frac{x}{s_k} + q_0\Big), \quad q \in [0, \text{maxq}]$$

          * Dequantize:

            $$\hat{x} = (q - q_0) s_k$$

          * Error $e = \hat{x} - x$.

      * Loss per candidate:

          * If `IS_L2_NORM` (i.e. $p = 2$):

            $$L_k = \sum e^2$$

          * Else general $L^p$ via:

            $$L_k = \sum |e|^p$$

3.  Pick the candidate $s_k$ with minimal loss and update `log2_scale_fp` in qmeta.

This search lives entirely in registers with warp-level reductions and constant-memory lookups for the grid `p`.

-----

## 7\. Python GPTQ Core (`forge/gptq.py`)

### 7.1 `GPTQ.build_quant_grid(...)`
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
Steps:

1.  Validate shapes and params.

2.  Compute padding and `num_groups`.

3.  Reshape to `x_groups: (C * num_groups, group_size)`.

4.  If CUDA:

      * Call `build_group_meta_packed_cuda`.
      * If `mode == "mse"`, build candidate grid `p` on GPU and call `mse_scale_groups_packed_cuda`.

5.  If CPU:

      * Use reference implementations:

          * `_find_quantization_meta_groups(...)`,
          * `_mse_scale_groups(...)`,
          * `_encode_qmeta_groups(...)`.

6.  Reshape qmeta to `(C, num_groups, 4)` and return `(qmeta, maxq, pad)`.

### 7.2 Reference Solver: `GPTQ.solver(...)`
```
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
Let:

  * $C =$ first dimension of `weight` (GPTQ coordinate dimension).
  * $R =$ second dimension.
  * $G =$ `num_groups`.

Algorithm (row-wise GPTQ, groupwise quantization):

1.  Convert:

      * `W = weight.to(float32)`.
      * `Hinv = hessian_inv.to(float32)`.
      * Allocate `qweight` as `(C, R)` uint8.

2.  For each row $j = 0 \ldots C-1$:

    1.  Extract the row’s group metadata:
    ```
      qmeta_row = qmeta[j]              # (G, 4)
          scale_row, qzero_row = decode_qmeta_groups(qmeta_row)  # (G,), (G,)
    ```
    2.  For each group index $g$:

          * Column range:

            $$\text{start} = g \cdot \text{group_size}, \quad \text{end} = \min(\text{start} + \text{group_size}, R)$$

          * Group scale $s_{j,g}$, zero-point $q_{0,j,g}$.

          * Slice:

            $$x = W[j, \text{start}:\text{end}]$$

          * Quantize:

            $$q = \text{clip}\Big(\text{round}\big(\frac{x}{s_{j,g}} + q_{0,j,g}\big), 0, \text{maxq}\Big)$$

          * Dequantize:

            $$\hat{x} = (q - q_{0,j,g}) \cdot s_{j,g}$$

          * Error:

            $$e_{g} = \hat{x} - x$$

          * Write back:

              * `W[j, start:end] = hat_x`,
              * `qweight[j, start:end] = q`.

    3.  Concatenate group errors into one error vector $e_j \in \mathbb{R}^{R}$.

    4.  **Error propagation (row-wise GPTQ):**

          * Take `h_tail = Hinv[j, j+1:]`.

          * Update later rows:

            $$W[j+1:, :] \mathrel{+}= h_{\text{tail}}^\top \otimes e_j$$

            i.e.

            $$W[r, :] \leftarrow W[r, :] + H^{-1}_{j,r} \cdot e_j \quad \text{for } r = j+1, \dots, C-1$$

3.  Return `qweight`.

This is the direct analogue of MoE-Quant’s Triton implementation, but using qmeta4 and explicit group loops.

### 7.3 Planned CUDA Solver (Phase-2)

The fused CUDA solver kernel will:

  * Take as input:

      * `weight (C, R)` (or a block of rows),
      * `hessian_inv (C, C)` (or its Cholesky factor),
      * `qmeta_bytes (C, G, 4)`,
      * `maxq`, `group_size`, `bits`.

  * For each row `j` in a block:

    1.  Decode that row’s qmeta into registers:

          * Precompute $s_{j,g}$, $1/s_{j,g}$, $q_{0,j,g}$ per group.

    2.  Quantize that row group-wise, producing `e_j` in registers/shared memory.

    3.  Use `Hinv[j, j+1:block_end]` to update the rest of the block (outer product) and a matmul to update rows outside the block — the same structure as MoE-Quant’s `gptq_loop`.

The **math** is identical to the Python reference; only the implementation changes.

-----

## 8\. Why 4Bit Forge’s Solver Is “True” to GPTQ

We want 4Bit Forge to produce the **same end result** as Algorithm 1, while leveraging:

  * groups,
  * qmeta4 packing,
  * log2 Q8.8 storage.

### 8.1 Transpose and axis alignment

Original GPTQ:

  * $W \in \mathbb{R}^{d_{\text{row}} \times d_{\text{col}}}$.
  * $H^{-1} \in \mathbb{R}^{d_{\text{col}} \times d_{\text{col}}}$.
  * Loop over *columns* $j \in \{0, \ldots, d_{\text{col}}-1\}$.

4Bit Forge / MoE-Quant:

  * Work with `weight_t` = $W^\top$ of shape `(C, R)` where $C = d_{\text{col}}$, $R = d_{\text{row}}$.
  * Hessian inverse `Hinv` has shape `(C, C)`.
  * Loop over **rows** `j` of `(C, R)`.

Mathematically this is just changing coordinates:

  * Column $j$ of the original $W$ is row $j$ of `weight_t`.

  * The GPTQ update:

    $$W_{:,k} \leftarrow W_{:,k} - e_j \cdot H^{-1}_{j,k}$$

    becomes:

    $$W^\top_{k,:} \leftarrow W^\top_{k,:} + H^{-1}_{j,k} \cdot e_j$$

    which is exactly what our row-wise update does.

So we are still implementing Algorithm 1, just on $W^\top$ instead of $W$.

### 8.2 Groupwise quantization vs per-element scales

In Algorithm 1, the quantizer `quant(·)` for column $j$ is abstract; it could be:

  * per-element scales,
  * per-column scale,
  * or per-group scale.

MoE-Quant uses **groupwise** scales but stores them expanded:

  * For each row `j` and group index `g`, they compute a scalar $s_{j,g}$ and zero-point $q_{0,j,g}$.
  * Then they broadcast these to `scale[j, :]` and `qzero[j, :]` so that the quantizer still acts **groupwise**, but the data structure is `(C, R)`.

4Bit Forge keeps the *same quantizer*, but:

  * Stores $s_{j,g}$ and $q_{0,j,g}$ directly as qmeta4,
  * Decodes them on the fly inside the solver kernel,
  * Applies them over the same contiguous ranges of columns (`group_size`).

So, mathematically, the quantization step:

$$Q_{:,j} = \text{quant}(W_{:,j})$$

is identical between MoE-Quant and 4Bit Forge — only the **representation** of the scales differs.

### 8.3 Q8.8 log2 storage and correctness

QMeta4 stores:

  * $\tilde{\ell}_{j,g} \approx \log_2(s_{j,g}) \cdot 256$ in `int16`,
  * decodes back as $\hat{s}_{j,g} = 2^{\tilde{\ell}_{j,g}/256}$.

This is a **reparameterization** of `scale`, not a change in the optimization problem. The only difference is a tiny relative error between the float scale used by MoE-Quant and the decoded scale used by Forge:

$$\frac{|\hat{s}_{j,g} - s_{j,g}|}{s_{j,g}} \ll 1$$

For typical ranges, this error is on the order of $10^{-3}$, which is dwarfed by 4-bit quantization noise. So:

  * The objective (minimize local quadratic loss using GPTQ) is unchanged.
  * The solver just uses a very slightly perturbed scale (well within numerical noise).

### 8.4 Blocked updates and MoE-Quant parity

Algorithm 1 uses blocksize $B$ and splits updates into:

  * *Within-block* updates during the inner loop,
  * *Outer-block* updates at the block end.

MoE-Quant copies this structure; 4Bit Forge’s planned Phase-2 solver does the same:

  * We still iterate over rows $j$ inside blocks,
  * Quantize each row,
  * Accumulate block errors,
  * Use a matmul to update rows outside the block.

As long as we:

  * Use the same block ordering,
  * Use the same Hessian inverse $H^{-1}$,
  * And define the same quantizer per row/group,

we are mathematically implementing Algorithm 1, just with more efficient metadata handling.

-----

## 9\. Mapping to MoE-Quant Components

Quick map from MoE-Quant to 4Bit Forge:

| MoE-Quant Component                   | 4Bit Forge Component                            | Status |
| ------------------------------------- | ----------------------------------------------- | ------ |
| `GPTQ.update(input)`                  | External Hessian builder                        | ⬜      |
| `quantization_pre_step()`             | External regularization + inversion             | ⬜      |
| `quant_utils.get_quantization_grid()` | `GPTQ.build_quant_grid(...)` → `(qmeta4, maxq)` | ✅      |
| Triton `mse_scale(...)`               | `mse_scale_groups_packed_cuda(...)`             | ✅      |
| `gptq_loop` (Triton)                  | `GPTQ.solver(...)` (PyTorch ref) / CUDA solver  | ✅ / ⬜  |

So 4Bit Forge drops into the “inner engine” of MoE-Quant with minimal glue.

-----

## 10\. Roadmap (Phases 2 & 3)

### Phase-2: Fused CUDA GPTQ Solver

Goals:

  * Implement a CUDA kernel that mirrors MoE-Quant’s `gptq_loop` but:

      * Consumes qmeta4 instead of `(C, R)` scale/qzero grids.
      * Decodes log2-Q8.8 scales into registers per row/group.
      * Uses warp/block-level tiling on `Hinv` and `W`.

  * Add a `use_cuda_solver=True` path in `GPTQ.solver(...)`.

Expected wins:

  * Less bandwidth (we never load giant `scale` / `qzero` tensors).
  * Better cache locality in the solver (group metadata is tiny; all hot data is `W` + `Hinv`).
  * Clean CUDA Graph state: only `qmeta4`, `Hinv`, and `W` need to live in the graph.

### Phase-3: Deeper qmeta4 / group-aware tricks (optional)

This phase is **exploratory** and can be skipped without breaking correctness:

  * Potential lines:

    1.  **Group-aware scheduling:**

          * Arrange the GPTQ coordinate order so that groups with similar scales or Hessian structure are processed together.
          * Could help with numerical stability and cache reuse.

    2.  **Hessian structure vs groups:**

          * Exploit block structure in $H^{-1}$ (if present) that aligns with quantization groups.
          * E.g., approximate off-group couplings, or compress $H^{-1}$ in a way that plays nicely with qmeta4.

    3.  **Preconditioning in log-scale space:**

          * Since qmeta4 stores $\log_2(s)$, one could do tiny local adjustments in log space (e.g. bias per channel) without touching the kernel interface.

Crucially, all Phase-3 ideas must **keep the GPTQ objective intact** (local quadratic minimization with respect to $H^{-1}$); they’re allowed to approximate *how* we get there, not *what* we’re optimizing.

-----

## 11\. Integration Pattern (Dense & MoE Layers)

For a dense `nn.Linear`:

1.  Transpose weights:
  ```
    W = layer.weight.data              # (d_out, d_in)
        W_t = W.transpose(0, 1).contiguous()  # (C = d_in, R = d_out)
  ```
2.  Build or load `Hinv` of shape `(C, C)`.

3.  Build qmeta4:
  ```
  gptq = GPTQ(group_size=128, bits=4, symmetric=True)
      qmeta, maxq, pad = gptq.build_quant_grid(
          W_t, group_size=128, bits=4,
          symmetric=True, mode="mse",
          quant_max_shrink=0.2, quant_n_grid=100, quant_norm=2.4,
      )
  ```
  4. Solve GPTQ:
  ```
  qweight_t = gptq.solver(
        weight=W_t,
        hessian_inv=Hinv,
        qmeta=qmeta,
        maxq=maxq,
        group_size=128,
        bits=4,
    )
  ```
  5.  Transpose back for storage/runtime:
  ```
  qweight = qweight_t.transpose(0, 1).contiguous()  # (d_out, d_in)
  ```
  For MoE MLPs, repeat this per expert with expert-specific Hessians.

---

## 12. 4Bit-Forge GPTQ Solver (qmeta4-First, Maximum Leverage)

This section describes the **target GPTQ solver design** for 4Bit-Forge that fully exploits:

* Groupwise quantization (`group_size`),
* Packed **qmeta4** (`QMetaPacked`),
* Log2 Q8.8 encoding of scales,
* Warp-level primitives: **broadcast** and **butterfly reductions**.

The goal is to stay **mathematically identical** to MoE-Quant’s GPTQ loop, while changing how metadata is represented, loaded, and applied.

---

### 12.1 Core Principle — True qmeta4 Leverage

We treat `QMetaPacked` / qmeta4 as the **single source of truth for quantization metadata**:

* We do **not** materialize `(C, R)` or `(C, G)` float `scale` / `qzero` tensors as full grids.
* `scale`, `inv_scale`, `qzero`, and `maxq` are **decoded per group** directly in registers (or shared memory) from a 4-byte packed struct.
* Warps cooperate to decode and reuse that metadata across the whole block.

C++ struct:

```cpp
struct QMetaPacked {
    int16_t  log2_scale_fp;  // Q8.8 fixed-point log2(scale)
    uint8_t  qzero;          // zero-point (0..255)
    uint8_t  flags;          // bitfield; bit0 = symmetric, others reserved
};
static_assert(sizeof(QMetaPacked) == 4, "QMetaPacked must be 4 bytes.");
```

When viewed as `uint32_t`:

* Bits [15:0]  → `log2_scale_fp` (Q8.8),
* Bits [23:16] → `qzero` (uint8),
* Bits [31:24] → `flags`.

#### 12.1.1 Bandwidth Comparison (Illustrative, `group_size = 128`)

Assume:

* MoE-Quant-style solver loads **per-element** `scale` and `qzero` (e.g. FP16 tensors with shape `(C, R)`),
* 4Bit-Forge solver loads **per-group** `QMetaPacked` (4 bytes per group).

For a row `j`:

| Layer type            | C    | R     | G (= ceil(R/128)) | Approx. MoE-Quant meta load per row (scale+qzero, fp16) | 4Bit-Forge qmeta load per row | Ratio |
| --------------------- | ---- | ----- | ----------------- | ------------------------------------------------------- | ----------------------------- | ----- |
| Attn Out (dense)      | 7168 | 7168  | 56                | ~14 KB                                                  | 56 × 4 B = 224 B              | 64×   |
| Shared Expert Up/Gate | 7168 | 18432 | 144               | ~36 KB                                                  | 144 × 4 B = 576 B             | 64×   |
| Routed Expert Up/Gate | 7168 | 2048  | 16                | ~4 KB                                                   | 16 × 4 B = 64 B               | 64×   |
| MLA KV Up (wide)      | 512  | 32768 | 256               | ~64 KB                                                  | 256 × 4 B = 1 KB              | 64×   |

So once we push this into a CUDA solver, **metadata bandwidth becomes negligible** relative to Hessian + weight traffic.

---

### 12.2 Final Decode Pattern (CUDA, log2 Q8.8)

We use log2 Q8.8 to cheaply obtain both `scale` and `inv_scale` per group.

```cpp
__constant__ float c_inv256 = 1.0f / 256.0f;
__constant__ float c_half   = 0.5f;

__device__ __forceinline__ void decode_qmeta(
    uint32_t packed,
    float&   scale,
    float&   inv_scale,
    float&   qzero_f,
    float&   maxq_g,
    uint8_t  global_bits  // e.g. 4
) {
    // Layout: [15:0] log2_scale_fp (Q8.8), [23:16] qzero, [31:24] flags
    int16_t log2_q88 = static_cast<int16_t>(packed & 0xFFFFu);
    uint8_t qzero_u8 = static_cast<uint8_t>((packed >> 16) & 0xFFu);
    uint8_t flags    = static_cast<uint8_t>(packed >> 24);

    // log2(scale) = log2_q88 / 256
    float log2_scale = __int2float_rn(log2_q88) * c_inv256;
    scale     = __exp2f(log2_scale);
    inv_scale = __exp2f(-log2_scale);  // cheap inverse via negative exponent

    // Bits-per-group (future-proof; current implementation uses global_bits)
    uint8_t bits_g = global_bits;
    // Reserved bits for per-group bits if needed later:
    // if (flags & 0x02) bits_g = 3;
    // if (flags & 0x04) bits_g = 5;

    int maxq_i = (1 << bits_g) - 1;
    maxq_g = static_cast<float>(maxq_i);

    // Symmetric override: qzero = (maxq + 1)/2
    if (flags & 0x01) {
        qzero_u8 = static_cast<uint8_t>((maxq_g + 1.0f) * c_half);
    }
    qzero_f = static_cast<float>(qzero_u8);
}
```

No division, just integer ops + `exp2`.

---

### 12.3 Per-Element Quant/Dequant (Fused)

Given `x` (fp32), and decoded `(scale, inv_scale, qzero_f, maxq_g)` for group `g`:

```cpp
float biased = x * inv_scale + qzero_f;          // x / s + q0
int q = __float2int_rn(biased);                  // round-to-nearest-even

int maxq_i = static_cast<int>(maxq_g);
q = q < 0 ? 0 : (q > maxq_i ? maxq_i : q);       // clamp to [0, maxq]

float deq = __fmaf_rn(static_cast<float>(q), scale, -qzero_f * scale);
// deq = q * s - q0 * s = (q - q0) * s

float err = deq - x;                             // GPTQ error term
```

This is the same math MoE-Quant uses:

* Quant: `q = clamp(round(x / s + q0), 0, qmax)`
* Dequant: `y = (q - q0) * s`, `e = y - x`

We just exploit `inv_scale = 1/s` from the log2 representation.

---

### 12.4 Shared-Memory Block Decode (Per-Block Preload)

We process columns in blocks `i..i_end` (block size `B = i_end - i_start`), and pre-decode all qmeta for that block into shared memory once.

```cpp
// For a block of B rows, each with G groups
extern __shared__ float smem[];  // layout decided at launch

float* sm_inv_scale = smem;
float* sm_scale     = sm_inv_scale +  B * G;
float* sm_qzero_f   = sm_scale     +  B * G;
float* sm_maxq_g    = sm_qzero_f   +  B * G;

// Optional PackBoost-style padding to avoid bank conflicts:
// constexpr int STRIDE = ((G + 7) & ~7);  // round up to multiple of 8
// and index by row * STRIDE + g instead of row * G + g.

int block_width = i_end - i_start;  // B

for (int idx = threadIdx.x; idx < block_width * G; idx += blockDim.x) {
    int row_in_block = idx / G;     // 0..B-1
    int g            = idx % G;     // 0..G-1

    int j = i_start + row_in_block; // global row index

    // qmeta_flat: [C * G] as uint32_t, row-major in groups
    uint32_t packed = qmeta_flat[j * G + g];

    float scale, inv_scale, qzero_f, maxq_g;
    decode_qmeta(packed, scale, inv_scale, qzero_f, maxq_g, global_bits);

    int off = row_in_block * G + g;
    sm_inv_scale[off] = inv_scale;
    sm_scale    [off] = scale;
    sm_qzero_f  [off] = qzero_f;
    sm_maxq_g   [off] = maxq_g;
}
__syncthreads();
```

After this:

* Every warp processing row `j` in `[i_start, i_end)` reads its `(scale, inv_scale, qzero, maxq)` for all groups from shared memory,
* No extra global loads for metadata inside the block.

---

### 12.4.1 Warp Broadcast (for Hinv Scalars)

For GPTQ error propagation, each row update uses a scalar coefficient `alpha = Hinv[j, k]`.

We only need **one lane** in the warp to read that value from global memory; the rest can receive it via a **warp broadcast**:

```cpp
__device__ __forceinline__ float warp_broadcast(float v, int src_lane = 0) {
    return __shfl_sync(0xffffffff, v, src_lane);
}
```

Usage pattern inside the solver kernel (inner-block updates):

```cpp
// Assume warp processes row k (or a tile of rows) for columns [0, R)
int lane_id = threadIdx.x & 31;

// Only lane 0 reads Hinv scalar from global memory
float alpha = 0.0f;
if (lane_id == 0) {
    alpha = Hinv[j * C + k];   // Hinv[j, k]
}

// Broadcast to all lanes in this warp
alpha = warp_broadcast(alpha, 0);

// Now every lane can update its slice of W_solver[k, :]
for (int col = lane_id; col < R; col += warpSize) {
    float e_val = error_block[row_in_block * R + col];
    W_solver[k * R + col] += alpha * e_val;
}
```

This:

* Cuts `Hinv` loads from `warpSize` loads → **1 load per warp**,
* Keeps the math exactly the same as `W[k] += Hinv[j, k] * e`.

---

### 12.4.2 Butterfly Reductions (Local Warp Sums)

We also keep a PackBoost-style **butterfly reduction** utility for any per-warp sum operations needed inside the solver (e.g. norms, diagnostics, or small tile reductions):

```cpp
template <typename T>
__device__ __forceinline__ T butterflyReduceSum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

Possible uses in the solver:

* Computing per-row or per-group error norms inside the warp for diagnostics,
* Reducing partial dot-products when doing tiny GEMM-like updates inside the block (if we ever add that path),
* Maintaining consistency with the reduction patterns already used in `quant_grid.cu`.

Functionally, this doesn’t change GPTQ math; it just gives us a standard warp-sum primitive to use wherever we need a reduction.

---

### 12.5 Python Reference Solver (qmeta4-First, No Scale Grid Expansion)

The Python reference solver mirrors the CUDA logic but keeps everything explicit and debuggable.

#### 12.5.1 Packed View of qmeta

Assume:

* `qmeta` is `(C, G, 4)` with dtype `torch.uint8`, contiguous.

Per row `j`:

```python
# qmeta[j]: (G, 4) uint8
qmeta_row = qmeta[j].contiguous()
qmeta_packed = qmeta_row.view(torch.uint32).view(-1)  # (G,)
```

Then:

```python
log2_q88 = (qmeta_packed & 0xFFFF).to(torch.int16)          # (G,)
qzero_u8 = ((qmeta_packed >> 16) & 0xFF).to(torch.uint8)    # (G,)
flags    = (qmeta_packed >> 24).to(torch.uint8)             # (G,)

log2_scale = log2_q88.to(torch.float32) / 256.0
scale      = torch.exp2(log2_scale)                         # (G,)
inv_scale  = torch.exp2(-log2_scale)                        # (G,)

bits_g = torch.full_like(flags, bits, dtype=torch.int32)    # global bits for now
maxq_per_group = (1 << bits_g) - 1                          # (G,)

qzero = qzero_u8.to(torch.float32)                          # (G,)

sym_mask = (flags & 0x01) != 0
sym_qzero = (maxq_per_group.to(torch.float32) + 1.0) * 0.5
qzero = torch.where(sym_mask, sym_qzero, qzero)             # (G,)
```

#### 12.5.2 Groupwise Quantization in the Reference Solver

Inside `GPTQ.solver(...)`, for row `j`:

```python
# W_solver: (C, R) float32 working buffer
# qweight:  (C, R) uint8

for g in range(num_groups):
    start = g * group_size
    end   = min(start + group_size, R)
    if start >= R:
        break

    s      = scale[g]                  # scalar
    inv_s  = inv_scale[g]
    z      = qzero[g]
    maxq_g = maxq_per_group[g].float()

    x = W_solver[j, start:end]         # (group_len,)

    biased = x * inv_s + z
    q = torch.round(biased)
    q = torch.clamp(q, 0.0, maxq_g)    # still float

    y = (q - z) * s
    e = y - x

    W_solver[j, start:end] = y
    qweight[j, start:end]  = q.to(torch.uint8)
    error_block[row_in_block, start:end] = e
```

Error propagation follows the standard GPTQ logic:

* **Inside the block:**

  ```python
  for k in range(j + 1, i_end):
      W_solver[k, :] += hessian_inv[j, k] * error_block[row_in_block, :]
  ```

* **Tail matmul after the block:**

  ```python
  if i_end < C:
      delta = hessian_inv[i_start:i_end, i_end:].T @ error_block[:(i_end - i_start), :]
      W_solver[i_end:, :] += delta
  ```

This is exactly the same algorithm as MoE-Quant’s `gptq_loop`:

* Same column/block order,
* Same error term `e = y - x`,
* Same propagation `W ← W + H⁻¹ e`.

The only differences are:

1. Metadata is **qmeta4**, decoded on the fly instead of pre-expanded `scale`/`qzero` grids.
2. On CUDA, we lean on:

   * Shared-memory block decode of qmeta,
   * **Warp broadcast** for Hessian scalars,
   * **Butterfly reductions** where we need warp-sum behavior.

Those are pure implementation wins; the math stays faithful to GPTQ.
