# 4Bit Forge: vLLM AWQ W4 Quantization Specification

This document defines the exact format required to produce AWQ-quantized models compatible with vLLM's optimized kernels. It is based on a reverse-engineering of the vLLM `v0.6.x` codebase.

---

## 1. vLLM Codebase Reference

The following files in the vLLM repository dictate the AWQ format:

1.  **Configuration & Loading:**
    *   `vllm/model_executor/layers/quantization/awq.py`
    *   `AWQConfig`: Defines expected config keys (`quant_method="awq"`, `bits=4`, `group_size=128`, `zero_point=True`).
    *   `AWQLinearMethod.create_weights`: Defines the storage shape of tensors.

2.  **Kernel Implementation (The Decoder):**
    *   `vllm/csrc/quantization/awq/gemm_kernels.cu`: The main GEMM kernel (`awq_gemm`).
    *   `vllm/csrc/quantization/awq/dequantize.cuh`: The bit-unpacking logic (`dequantize_s4_to_fp16x2`).

3.  **MoE Handling:**
    *   `vllm/model_executor/layers/fused_moe/layer.py`: Handles loading individual expert weights from checkpoints.

---

## 2. Exact W4 Tensor Layout

For any Linear layer (Dense, Attention Projection, or Expert Projection) with logical dimensions:
*   **Input Features ($IC$):** `in_features`
*   **Output Features ($OC$):** `out_features`

vLLM expects three tensors.

### 2.1 `qweight` (Quantized Weights)
This tensor stores the 4-bit weight indices.

*   **Dtype:** `torch.int32`
*   **Shape:** `[in_features, out_features // 8]`
    *   *Note:* This dimension order is effectively transposed compared to PyTorch's standard `[out_features, in_features]`.
*   **Packing:** "Interleaved"
    *   One `int32` element packs **8 consecutive output weights** for a single input channel.
    *   Logical weights: $w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7$ (corresponding to output columns $c, c+1, \dots, c+7$).
    *   **Bit Layout:**
        *   **Byte 0:** `[w2][w0]` (Bits 4-7: $w_2$, Bits 0-3: $w_0$)
        *   **Byte 1:** `[w6][w4]` (Bits 12-15: $w_6$, Bits 8-11: $w_4$)
        *   **Byte 2:** `[w3][w1]` (Bits 20-23: $w_3$, Bits 16-19: $w_1$)
        *   **Byte 3:** `[w7][w5]` (Bits 28-31: $w_7$, Bits 24-27: $w_5$)

### 2.2 `scales` (Quantization Scales)
*   **Dtype:** `torch.float16`
*   **Shape:** `[in_features // group_size, out_features]`
*   **Order:** Row-major.
    *   `scales[g, c]` is the scale factor for the group $g$ and output channel $c$.
*   **Group Size:** Typically 128.

### 2.3 `qzeros` (Zero Points)
*   **Dtype:** `torch.int32`
*   **Shape:** `[in_features // group_size, out_features // 8]`
*   **Packing:** **Identical to `qweight`**.
    *   The zero points are packed using the same interleaved bit-pattern as the weights.
    *   Note: AWQ zero points are typically added to the unpacked weight: $W_{fp16} = (W_{int4} - Z_{int4}) \times S_{fp16}$.

---

## 3. Packing Algorithm (Python Reference)

Use this function to pack standard Int4 tensors (values 0-15) into the vLLM format.

```python
import torch

def pack_vllm_awq(int4_tensor: torch.Tensor) -> torch.Tensor:
    """
    Packs a tensor of shape [in_features, out_features] containing 
    int4 values (0-15) into the vLLM AWQ interleaved format.
    
    Args:
        int4_tensor: Tensor of shape [in_features, out_features], 
                     dtype=torch.int32 (or int8).
    
    Returns:
        Packed tensor of shape [in_features, out_features // 8], dtype=torch.int32.
    """
    K, N = int4_tensor.shape
    assert N % 8 == 0, "Output dimension N must be divisible by 8"
    
    # 1. Reshape to isolate groups of 8 output channels
    # Shape: [K, N // 8, 8]
    groups = int4_tensor.reshape(K, N // 8, 8).to(torch.int32)
    
    # 2. Extract columns for interleaved packing
    w0 = groups[:, :, 0]
    w1 = groups[:, :, 1]
    w2 = groups[:, :, 2]
    w3 = groups[:, :, 3]
    w4 = groups[:, :, 4]
    w5 = groups[:, :, 5]
    w6 = groups[:, :, 6]
    w7 = groups[:, :, 7]
    
    # 3. Bit manipulation
    # Pattern: 0, 2, 4, 6, 1, 3, 5, 7
    packed = (w0 & 0xF)       | \
             ((w2 & 0xF) << 4)  | \
             ((w4 & 0xF) << 8)  | \
             ((w6 & 0xF) << 12) | \
             ((w1 & 0xF) << 16) | \
             ((w3 & 0xF) << 20) | \
             ((w5 & 0xF) << 24) | \
             ((w7 & 0xF) << 28)
             
    return packed.to(torch.int32)
```

---

## 4. Model-Specific Layer Formats

### 4.1 DeepSeek-Math-V2 (MoE + MLA)

vLLM identifies layers by their name in the `state_dict`. You must produce keys matching this structure.

#### **Config Requirements**
The model folder must contain a `quantize_config.json` (or merged into `config.json`):
```json
{
  "quantization_config": {
    "quant_method": "awq",
    "bits": 4,
    "group_size": 128,
    "zero_point": true,
    "modules_to_not_convert": []
  }
}
```

#### **Attention (MLA)**
Standard projections. Example for layer `0`:

*   `model.layers.0.self_attn.q_a_proj.qweight`
*   `model.layers.0.self_attn.q_b_proj.qweight`
*   `model.layers.0.self_attn.kv_a_proj_with_mqa.qweight`
*   `model.layers.0.self_attn.kv_b_proj.qweight`
*   `model.layers.0.self_attn.o_proj.qweight`

*(Each has corresponding `.scales` and `.qzeros`)*

#### **Mixture of Experts (MoE)**
Experts are saved **individually**. vLLM fuses them at load time.
For each expert $e$ (from 0 to $N_{experts}-1$):

*   **Gate:** `model.layers.0.mlp.experts.{e}.gate_proj.qweight`
*   **Up:** `model.layers.0.mlp.experts.{e}.up_proj.qweight`
*   **Down:** `model.layers.0.mlp.experts.{e}.down_proj.qweight`

#### **Shared Experts**
Saved as a dense MLP block.

*   `model.layers.0.mlp.shared_experts.gate_proj.qweight`
*   `model.layers.0.mlp.shared_experts.up_proj.qweight`
*   `model.layers.0.mlp.shared_experts.down_proj.qweight`

---

## 5. Proposed Pipeline: "4Bit Forge" (PackBoost Enhanced)

To achieve state-of-the-art quantization speed, "4Bit Forge" should replace the standard multi-pass quantization pipeline with a single fused kernel, adopting strategies from **PackBoost**.

### 5.1 Current "Naive" Pipeline (Standard vLLM/AutoAWQ)
1.  **VRAM Load:** Read FP16 weights.
2.  **Quantize:** Compute Int8 tensor -> **Write to VRAM**.
3.  **Permute:** Read Int8, Reorder `0,2,4,6...` -> **Write to VRAM**.
4.  **Pack:** Read Permuted Int8, Shift & OR -> **Write Int32 to VRAM**.
    *   *Inefficiency:* 3-4x memory bandwidth usage due to intermediate writes.

### 5.2 "4Bit Forge" Pipeline
**Single Fused CUDA Kernel:** `pack_awq_fused<<<...>>>`

1.  **Warp-Aligned Load (Coalesced):**
    *   Calculate grid strides to align with 32-thread warps.
    *   Load a `32x8` (or `32x32`) tile of FP16 weights directly into Registers/Shared Memory.
2.  **Fused Quantization (Compute-Bound):**
    *   Apply scale/zero math in registers.
    *   Convert to 4-bit integers.
3.  **Skewed Shared Memory Transpose (Bank-Conflict Free):**
    *   Write quantized nibbles to Shared Memory using skewed indices: `smem[row][(col+row)%32]`.
    *   This prevents bank conflicts when threads read columns to interleave.
4.  **Register Packing:**
    *   Read back 8 nibbles in the required `0,2,4,6,1,3,5,7` order.
    *   Bitwise-OR them into a single `uint32` register.
5.  **Coalesced Store:**
    *   Write the final `qweight` directly to VRAM.

**Result:** 1 Read + 1 Write. Max theoretical throughput.

---

## 6. Implementation Checklist

*   [ ] **Transpose:** Ensure weights are logical `[In, Out]` before packing.
*   [ ] **Group Size:** Verify `scales` shape matches `In // 128`.
*   [ ] **Interleaving:** Verify packing order `0, 2, 4, 6, 1, 3, 5, 7` (Low/High nibbles).
*   [ ] **MoE Naming:** Verify expert keys follow `experts.{id}.{proj}`.
*   [ ] **Dtypes:** `qweight`/`qzeros` (Int32), `scales` (Float16).
