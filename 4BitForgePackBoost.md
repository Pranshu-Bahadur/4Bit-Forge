# 4BitForgePackBoost.md

#### 1. High-Level Overview

**PackBoost** is a highly optimized Gradient Boosted Decision Tree (GBDT) library built for NVIDIA GPUs. At its core, it aims to maximize memory bandwidth utilization and arithmetic intensity by using custom bit-packed data structures. It employs techniques like "bit-plane encoding" to store features compactly and "warp-aggregated histograms" to compute gradients efficiently without atomic collisions. It bypasses standard libraries (like Thrust or CUB) in favor of hand-written CUDA kernels that manage warp-level parallelism, shared memory bank conflicts, and register pressure explicitly.

**4BIT FORGE** is a project aiming to provide a high-performance quantization and inference path for DeepSeek-Math-V2 on vLLM.
*   **Immediate Goal:** Implement a quantizer that produces the exact `qweight` (interleaved int32), `scales` (fp16), and `qzeros` (interleaved int32) format expected by vLLM's AWQ kernels.
*   **Future Goal:** Implement custom W4A16 GEMM kernels that outperform the existing vLLM AWQ/Marlin kernels by leveraging the same low-level bit-manipulation and memory layout tricks found in PackBoost.

---

#### 2. Kernel Techniques in PackBoost

| Technique | Location | Concept & Rationale |
| :--- | :--- | :--- |
| **Skewed Shared Memory Transpose** | `packboost/cuda/encode_cuts.cu` (`_encode_cuts`) | **Concept:** When loading a tile (e.g., 32x32) into shared memory, indices are rotated: `sm[wi][(k + wi) & 31]`.<br>**Rationale:** Prevents bank conflicts. If threads in a warp access the same column index `k`, they hit the same memory bank (serialized access). Skewing ensures all 32 threads access distinct banks simultaneously, maximizing L1/SMEM bandwidth. |
| **Register-Level Bit Packing** | `packboost/cuda/encode_cuts.cu` | **Concept:** The kernel constructs 32-bit integers from 4 separate bit-planes entirely in registers using bitwise ORs (`|=`) before writing to global memory.<br>**Rationale:** Dramatically reduces global memory traffic. Instead of 4 small writes (one per bit-plane), it performs 1 coalesced 32-bit write. Registers are the fastest memory hierarchy; utilizing them for "assembly" avoids VRAM bottlenecks. |
| **Warp-Aligned Grid Strides** | `packboost/cuda/repack.cu` | **Concept:** Loop strides are calculated as `total_threads` (gridDim * blockDim), but explicitly rounded to multiples of 32.<br>**Rationale:** Guarantees coalesced global memory access. Even if the matrix width is odd (e.g., `N=31999`), the kernel always reads/writes full 128-byte cache lines. "Tail" masking happens in registers, not by fragmenting memory requests. |
| **Warp-Aggregated Atomics** | `packboost/cuda/h.cu` (`_h_sm`) | **Concept:** Before adding to a global histogram, threads in a warp combine their updates using `__shfl_sync` (butterfly reduction) or shared memory atomics.<br>**Rationale:** Reduces global atomic contention. Instead of 32 threads atomically adding to the same address (high serialization), one leader thread adds the aggregate sum once. |
| **Templated Kernel Dispatch** | `packboost/cuda/repack.cu` | **Concept:** Kernels accept `template <typename PackedT>` to handle `uint16`, `uint32`, or `uint64` storage formats with the same logic.<br>**Rationale:** Code reuse and flexibility. Allows the same kernel to support different quantization precisions or packing densities without duplicating logic. |

---

#### 3. How Each Technique Could Enhance 4BIT FORGE

##### **A. Efficient AWQ Packing (The "Interleaved" Problem)**
vLLM requires weights to be packed in a non-linear order (`0, 2, 4, 6, 1, 3, 5, 7`).
*   **PackBoost Technique:** Skewed Shared Memory Transpose.
*   **Application:**
    1.  Load a `32x8` tile of FP16 weights into SMEM using coalesced reads.
    2.  Write to SMEM using the **skewed pattern** to avoid bank conflicts.
    3.  Read back from SMEM in the required interleaved order (`row, col` -> `row, perm[col]`). Because of the skew, this "strided" read will still be conflict-free.
    4.  Pack into `int32` in registers.
    5.  Write `int32` to global memory.
*   **Result:** A single-pass quantization kernel that runs at near-copy speeds, unlike current multi-pass permute-then-pack approaches.

##### **B. W4 GEMM Kernel Design**
*   **PackBoost Technique:** Warp-Aggregated Atomics & Register Packing.
*   **Application:**
    *   **Dequantization:** Load `qweight` (int32) into registers. Use bit-shifts (similar to PackBoost's bit-plane construction) to extract 8x4-bit values.
    *   **Fused Scale/Zero:** Perform `(w - z) * s` in registers immediately after unpacking.
    *   **Accumulation:** Use `wmma` (Tensor Cores) for the matmul. For the reduction (K-dimension), use **warp-shuffle** (butterfly reduction) inside the kernel to sum partial results across the warp before writing to global memory. This avoids the need for a separate reduction kernel.

##### **C. Handling Odd Shapes (Vocab Size / MoE Experts)**
DeepSeek-V2 has many experts with relatively small hidden dimensions or odd vocab sizes.
*   **PackBoost Technique:** Warp-Aligned Grid Strides.
*   **Application:**
    *   Write the W4 GEMM kernel to process tiles of `N=32` or `N=64`.
    *   Use the **grid-stride loop** pattern to handle arbitrary shapes. If `N % 32 != 0`, the loop logic naturally handles the boundary with predicate masking (`if col < N`), but the *memory access* remains 128-byte aligned (reading valid data + padding garbage, then masking). This prevents the drastic slowdown seen in standard kernels when dimensions aren't powers of 2.

##### **D. PyTorch Binding Strategy**
*   **PackBoost Pattern:** `kernels.cpp` defines `torch::Tensor` interfaces that cast to raw pointers and call CUDA launchers.
*   **Application:**
    *   Expose `forge_quantize_awq` and `forge_gemm_w4` as custom ops.
    *   Use the `PYBIND11_MODULE` structure to cleanly separate CPU checks (strides, dtypes) from the raw CUDA execution.
    *   **Parity Testing:** Keep a pure Python implementation of the "Interleaved" packing (from `4BitForgeSpecifications.md`) and use it in `tests/` to verify the CUDA kernel's output bit-for-bit.

---

#### 4. Proposed Kernel Architecture for 4BIT FORGE

**Kernel Name:** `awq_fused_quantize_interleaved`

**Memory Layout:**
*   **Input:** `Weights [K, N]` (FP16, Row Major)
*   **Output:** `QWeights [K, N//8]` (Int32, Interleaved), `Scales` (FP16), `QZeros` (Int32)

**Grid/Block Configuration:**
*   **Block:** `dim3(32, 4)` (128 threads, 4 warps). Each warp handles a `32x8` logical tile of the input matrix (32 rows, 8 columns).
*   **Grid:** Covered by the Warp-Aligned Grid Stride logic.

**Algorithm (Per Warp):**
1.  **Coalesced Load:** Warp loads 32 rows x 8 columns of FP16 weights.
    *   *Optimization:* Use `ld.global.v4` to load 128 bits (8x FP16) per thread if possible, or standard coalesced loads.
2.  **Statistics (Register):** Compute `min` and `max` across the group of 128 (if `K` dimension loop). Calculate `scale` and `zero`.
3.  **Quantize (Register):** `q = clamp(round(w / scale) + zero, 0, 15)`.
4.  **Skewed SMEM Store:**
    *   `smem[lane_id][(0 + skew) % 32] = q_col0`
    *   `smem[lane_id][(1 + skew) % 32] = q_col1`
    *   ...
5.  **Interleaved Load (Register):**
    *   Read 8 values from SMEM in the order `0, 2, 4, 6, 1, 3, 5, 7`.
    *   Because of step 4, these reads are bank-conflict free.
6.  **Bit Packing (Register):**
    *   `int32 out = w0 | (w2 << 4) | (w4 << 8) ...`
7.  **Store:** Write `out` to `QWeights`. Write `scale` and `zero` to their tensors.

---

#### 5. Actionable Todo List

*   [ ] **Step 1: Parity Reference (CPU):** Implement the python `pack_vllm_awq` function (from spec) and verify it produces identical files to a slow AutoAWQ run on a small layer.
*   [ ] **Step 2: Skeleton Kernel:** Set up the `setup.py` and `kernels.cpp` scaffolding (copied from PackBoost) to compile a dummy CUDA kernel that can be called from Python.
*   [ ] **Step 3: SMEM Transpose Prototype:** Implement a standalone CUDA kernel that just reads FP16, does the **Skewed Shared Memory Transpose**, and writes back to verify bank-conflict-free bandwidth.
*   [ ] **Step 4: Fused Quantizer:** Combine the quantization logic with the Transpose prototype. Verify correctness against the Python reference.
*   [ ] **Step 5: Benchmark:** Measure throughput (GB/s) of `4BIT FORGE` vs `AutoAWQ` packing. Expect >3x speedup.
*   [ ] **Step 6: Integration:** Wrap the kernel in a module that iterates over a DeepSeek checkpoint and saves the quantized tensors.
