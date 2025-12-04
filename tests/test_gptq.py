import math
import pytest
import torch

from forge.gptq import GPTQ

def has_cuda():
    return torch.cuda.is_available()

def has_fp8():
    return hasattr(torch, "float8_e4m3fn")

# ---------- helpers for tests ----------

def unpack_qmeta_tensor(qmeta, group_size, R):
    """
    Test helper to unpack (C, G, 4) qmeta back to (C, R) scale/qzero tensors.
    This simulates the broadcast logic needed for reference comparisons.
    """
    C, G, _ = qmeta.shape
    device = qmeta.device
    
    # 1. Decode (C, G)
    lo = qmeta[..., 0].to(torch.int16)
    hi = qmeta[..., 1].to(torch.int16)
    # Combine bytes: little-endian
    # Note: Torch bitwise ops on signed integers behave like C++ (2s complement)
    log2_q88 = (lo & 0x00FF) | (hi << 8)
    
    scale_g = torch.exp2(log2_q88.float() / 256.0) # (C, G)
    qzero_g = qmeta[..., 2].float()                # (C, G)

    # 2. Expand to (C, G * group_size)
    # View as (C, G, 1) -> Expand -> Reshape
    scale_expanded = scale_g.unsqueeze(-1).expand(C, G, group_size).reshape(C, -1)
    qzero_expanded = qzero_g.unsqueeze(-1).expand(C, G, group_size).reshape(C, -1)

    # 3. Crop padding
    scale = scale_expanded[:, :R].contiguous()
    qzero = qzero_expanded[:, :R].contiguous()
    
    return scale, qzero

def quantize_dequant(weight, scale, qzero, maxq):
    """
    Local helper to simulate actual affine quantization math:
      q = round(x / s + z).clamp(0, maxq)
      y = (q - z) * s
    All tensors are (C, R).
    """
    x = weight.to(torch.float32)
    s = scale.to(torch.float32)
    z = qzero.to(torch.float32)
    maxq_val = float(maxq.item())

    q = torch.round(x / s + z)
    q.clamp_(0.0, maxq_val)
    y = (q - z) * s
    return q, y

def reference_gptq_solver(weight, hessian_inv, scale, qzero, maxq):
    """
    Tiny pure-CPU reference to compare against GPTQ.solver.
    Takes unpacked scale/qzero (C, R).
    """
    assert weight.ndim == 2
    C, R = weight.shape
    W = weight.to(torch.float32).clone()
    Hinv = hessian_inv.to(torch.float32).clone()
    S = scale.to(torch.float32)
    Z = qzero.to(torch.float32)
    maxq_val = float(maxq.item())

    qweight = torch.empty_like(weight, dtype=torch.uint8)

    for j in range(C):
        x = W[j]
        s = S[j]
        z = Z[j]

        q = torch.round(x / s + z)
        q.clamp_(0.0, maxq_val)
        q_codes = q.to(torch.uint8)

        y = (q - z) * s
        e = y - x

        W[j].copy_(y)
        qweight[j].copy_(q_codes)

        if j + 1 < C:
            h_row_tail = Hinv[j, j + 1 :]
            # Update W[j+1:, :] += h_row_tail[:, None] @ e[None, :]
            W[j + 1 :] += h_row_tail.unsqueeze(1) * e.unsqueeze(0)

    return qweight, W

# ---------- tests for build_quant_grid ----------

@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [32, 128])
def test_build_quant_grid_shapes_cpu(bits, group_size):
    torch.manual_seed(0)
    C, R = 7, 257  # odd dims to test padding logic
    W = torch.randn(C, R, dtype=torch.float32, device="cpu")

    gptq = GPTQ()
    qmeta, maxq, pad = gptq.build_quant_grid(
        W,
        group_size=group_size,
        bits=bits,
        symmetric=False,
        mode="absmax",
    )

    num_groups = (R + group_size - 1) // group_size
    
    assert qmeta.shape == (C, num_groups, 4)
    assert qmeta.dtype == torch.uint8
    assert isinstance(maxq, torch.Tensor)
    assert maxq.dim() == 0
    assert maxq.item() == (2**bits - 1)
    
    expected_pad = (num_groups * group_size) - R
    assert pad == expected_pad

@pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
@pytest.mark.parametrize("bits", [4])
def test_build_quant_grid_cpu_vs_gpu_error(bits):
    """
    Compare CPU vs GPU grid builder indirectly by comparing quantization error.
    Note: GPU uses packed Q8.8 scales, CPU uses pure float32 emulation.
    Small precision diffs are expected due to Q8.8 encoding.
    """
    torch.manual_seed(0)
    C, R = 8, 256
    group_size = 128
    W_cpu = torch.randn(C, R, dtype=torch.float32, device="cpu")

    gptq = GPTQ()

    # 1. CPU path
    qmeta_cpu, maxq_cpu, _ = gptq.build_quant_grid(
        W_cpu,
        group_size=group_size,
        bits=bits,
        symmetric=False,
        mode="absmax",
    )
    # Unpack packed meta to full (C, R) maps
    scale_cpu, qzero_cpu = unpack_qmeta_tensor(qmeta_cpu, group_size, R)
    _, y_cpu = quantize_dequant(W_cpu, scale_cpu, qzero_cpu, maxq_cpu)
    mse_cpu = ((y_cpu - W_cpu) ** 2).mean().item()

    # 2. GPU path
    W_gpu = W_cpu.to("cuda")
    qmeta_gpu, maxq_gpu, _ = gptq.build_quant_grid(
        W_gpu,
        group_size=group_size,
        bits=bits,
        symmetric=False,
        mode="absmax",
    )
    qmeta_gpu = qmeta_gpu.cpu()
    maxq_gpu = maxq_gpu.cpu()
    
    scale_gpu, qzero_gpu = unpack_qmeta_tensor(qmeta_gpu, group_size, R)
    _, y_gpu = quantize_dequant(W_cpu, scale_gpu, qzero_gpu, maxq_gpu)
    mse_gpu = ((y_gpu - W_cpu) ** 2).mean().item()

    # The Q8.8 encoding error (~0.27%) is small enough that MSE should match closely
    assert math.isclose(mse_cpu, mse_gpu, rel_tol=0.05, abs_tol=1e-4)

@pytest.mark.parametrize("mode", ["absmax", "mse"])
def test_build_quant_grid_mse_does_not_increase_error(mode):
    """
    Check that MSE mode doesn't worsen error compared to absmax.
    """
    torch.manual_seed(123)
    C, R = 4, 128
    group_size = 128
    W = torch.randn(C, R, dtype=torch.float32)

    gptq = GPTQ()

    # Baseline absmax
    qmeta_abs, maxq_abs, _ = gptq.build_quant_grid(
        W,
        group_size=group_size,
        bits=4,
        symmetric=False,
        mode="absmax",
    )
    scale_abs, qzero_abs = unpack_qmeta_tensor(qmeta_abs, group_size, R)
    _, y_abs = quantize_dequant(W, scale_abs, qzero_abs, maxq_abs)
    mse_abs = ((y_abs - W) ** 2).mean().item()

    # MSE-refined
    qmeta_mse, maxq_mse, _ = gptq.build_quant_grid(
        W,
        group_size=group_size,
        bits=4,
        symmetric=False,
        mode="mse",
        quant_max_shrink=0.2,
        quant_n_grid=16,
        quant_norm=2.4,
    )
    scale_mse, qzero_mse = unpack_qmeta_tensor(qmeta_mse, group_size, R)
    _, y_mse = quantize_dequant(W, scale_mse, qzero_mse, maxq_mse)
    mse_mse = ((y_mse - W) ** 2).mean().item()

    # In practice mse_mse <= mse_abs; allow tiny numerical wiggle
    assert mse_mse <= mse_abs + 1e-5

# ---------- tests for solver ----------

@pytest.mark.parametrize("C,R", [(4, 32), (6, 64)])
def test_solver_matches_reference_cpu(C, R):
    """
    Compare GPTQ.solver (using packed qmeta) against pure reference (unpacked scale)
    for small matrices on CPU.
    """
    torch.manual_seed(0)
    W = torch.randn(C, R, dtype=torch.float32, device="cpu")
    
    # SPD-ish Hessian
    X = torch.randn(C, C, dtype=torch.float32)
    H = X @ X.T + 1e-3 * torch.eye(C, dtype=torch.float32)
    Hinv = torch.inverse(H)

    gptq = GPTQ()
    group_size = min(32, R)
    bits = 4

    # 1. Build grid -> qmeta
    qmeta, maxq, _ = gptq.build_quant_grid(
        W, group_size=group_size, bits=bits, symmetric=False, mode="absmax"
    )

    # 2. Unpack for reference solver
    scale_ref, qzero_ref = unpack_qmeta_tensor(qmeta, group_size, R)
    
    # 3. Run reference solver
    q_ref, W_ref = reference_gptq_solver(W, Hinv, scale_ref, qzero_ref, maxq)

    # 4. Run GPTQ solver (groupwise loop with packed meta)
    W_clone = W.clone()
    q_gptq = gptq.solver(
        weight=W_clone,
        hessian_inv=Hinv,
        qmeta=qmeta,
        maxq=maxq,
        group_size=group_size,
        bits=bits,
    )

    assert torch.equal(q_gptq, q_ref)
    assert torch.allclose(W_clone, W_ref, rtol=1e-5, atol=1e-6)

@pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_build_quant_grid_supports_fp16_bf16(dtype):
    """
    Ensure build_quant_grid works with various input dtypes on CUDA.
    """
    torch.manual_seed(42)
    C, R = 8, 256
    group_size = 128
    W = torch.randn(C, R, dtype=torch.float32, device="cuda").to(dtype)

    gptq = GPTQ()
    qmeta, maxq, _ = gptq.build_quant_grid(
        W,
        group_size=group_size,
        bits=4,
        symmetric=False,
        mode="mse",
        quant_n_grid=8,
    )

    num_groups = (R + group_size - 1) // group_size
    assert qmeta.shape == (C, num_groups, 4)
    assert qmeta.dtype == torch.uint8
    assert maxq.shape == torch.Size([])

@pytest.mark.skipif(not (has_cuda() and has_fp8()), reason="float8 or CUDA not available")
def test_build_quant_grid_supports_fp8_e4m3():
    """
    Smoke test: weight in float8_e4m3fn should not crash build_quant_grid.
    """
    torch.manual_seed(7)
    C, R = 4, 128
    W_fp32 = torch.randn(C, R, dtype=torch.float32, device="cuda")
    W_fp8 = W_fp32.to(torch.float8_e4m3fn)

    gptq = GPTQ()
    qmeta, maxq, _ = gptq.build_quant_grid(
        W_fp8,
        group_size=128,
        bits=4,
        symmetric=False,
        mode="absmax",
    )

    assert qmeta.shape == (C, 1, 4)
    assert maxq.item() == 2**4 - 1