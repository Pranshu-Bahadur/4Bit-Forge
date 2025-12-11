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

    NOTE: This uses the simpler decoding that matches the *grid builder*
    tests, not the full solver semantics. It’s meant only for approximate
    error comparison in build_quant_grid tests.
    """
    C, G, _ = qmeta.shape

    # 1. Decode (C, G) Q8.8 log2(scale)
    lo = qmeta[..., 0].to(torch.int16)
    hi = qmeta[..., 1].to(torch.int16)

    # Combine bytes: little-endian into signed int16
    log2_q88 = (lo & 0x00FF) | (hi << 8)

    scale_g = torch.exp2(log2_q88.float() / 256.0)  # (C, G)
    qzero_g = qmeta[..., 2].float()                 # (C, G)

    # 2. Expand to (C, G * group_size)
    scale_expanded = (
        scale_g.unsqueeze(-1)
        .expand(C, G, group_size)
        .reshape(C, -1)
    )
    qzero_expanded = (
        qzero_g.unsqueeze(-1)
        .expand(C, G, group_size)
        .reshape(C, -1)
    )

    # 3. Crop padding
    scale = scale_expanded[:, :R].contiguous()
    qzero = qzero_expanded[:, :R].contiguous()

    return scale, qzero


def quantize_dequant(weight, scale, qzero, maxq):
    """
    Local helper to simulate affine quantization:
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


# ---------- solver-specific helpers (match CUDA/CPU solver exactly) ----------


def _decode_qmeta_row_for_solver(qmeta_row: torch.Tensor, bits: int):
    """
    Decode a single row's qmeta: (G, 4) uint8 -> (scale_g, inv_scale_g, qzero_g, maxq_val)
    using the *same* logic as the CUDA / CPU solver:

    - Q8.8 log2(scale) with sign extension
    - per-row symmetric flag: qzero = (maxq + 1)/2 when flags & 1
    """
    meta = qmeta_row.to(torch.int32)  # (G, 4)

    lo = meta[:, 0]
    hi = meta[:, 1]
    log2_q88 = lo | (hi << 8)  # int16 stored in 2 bytes

    # Manual sign-extension from 16-bit to 32-bit
    log2_q88 = torch.where(log2_q88 >= 32768, log2_q88 - 65536, log2_q88)

    log2_scale = log2_q88.float() / 256.0
    scale_g = torch.exp2(log2_scale)
    inv_scale_g = torch.exp2(-log2_scale)

    qzero_u8 = meta[:, 2].float()
    flags = meta[:, 3]

    maxq_val = float((1 << bits) - 1)

    # Symmetric override: qzero = (maxq + 1)/2
    is_sym = (flags & 1) != 0
    sym_q0 = (maxq_val + 1.0) * 0.5
    qzero_g = torch.where(is_sym, sym_q0, qzero_u8)

    return scale_g, inv_scale_g, qzero_g, maxq_val


def reference_gptq_solver_from_qmeta(weight, hessian_inv, qmeta, group_size, bits):
    """
    Reference GPTQ solver that:
    - decodes qmeta exactly like the 4Bit-Forge solver
    - uses the same quantization math: x * inv_s + q0, clamp, (q - q0) * s
    - uses the same update rule: W[k] += Hinv[j, k] * e

    This is the "gold standard" comparison target for GPTQ.solver on both CPU and CUDA.
    """
    assert weight.ndim == 2
    C, R = weight.shape

    W = weight.to(torch.float32).clone()
    Hinv = hessian_inv.to(torch.float32).clone()

    qweight = torch.empty_like(weight, dtype=torch.uint8)

    num_groups = qmeta.size(1)

    for j in range(C):
        row_meta = qmeta[j]  # (G, 4)
        scale_g, inv_scale_g, qzero_g, maxq_val = _decode_qmeta_row_for_solver(
            row_meta, bits
        )

        if j + 1 < C:
            h_tail = Hinv[j, j + 1 :]  # (C - j - 1,)
        else:
            h_tail = None

        for g in range(num_groups):
            start = g * group_size
            if start >= R:
                break
            end = min(start + group_size, R)

            s = scale_g[g]
            inv_s = inv_scale_g[g]
            q0 = qzero_g[g]

            x = W[j, start:end]

            biased = x * inv_s + q0
            q = torch.round(biased)
            q.clamp_(0.0, maxq_val)

            y = (q - q0) * s
            e = y - x

            W[j, start:end] = y
            qweight[j, start:end] = q.to(torch.uint8)

            if h_tail is not None:
                # W[j+1:, cols] += h_tail[:, None] * e[None, :]
                W[j + 1 :, start:end] += h_tail.unsqueeze(1) * e.unsqueeze(0)

    return qweight, W


# ---------- tests for build_quant_grid ----------


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("impl", ["cuda", "triton"])
def test_build_quant_grid_shapes_cpu(bits, group_size, impl):
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
        impl=impl,
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
@pytest.mark.parametrize("impl", ["cuda", "triton"])
def test_build_quant_grid_cpu_vs_gpu_error(bits, impl):
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
        impl=impl,
    )
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
        impl=impl,
    )
    qmeta_gpu = qmeta_gpu.cpu()
    maxq_gpu = maxq_gpu.cpu()

    scale_gpu, qzero_gpu = unpack_qmeta_tensor(qmeta_gpu, group_size, R)
    _, y_gpu = quantize_dequant(W_cpu, scale_gpu, qzero_gpu, maxq_gpu)
    mse_gpu = ((y_gpu - W_cpu) ** 2).mean().item()

    # The Q8.8 encoding error (~0.27%) is small enough that MSE should match closely
    assert math.isclose(mse_cpu, mse_gpu, rel_tol=0.05, abs_tol=1e-4)


@pytest.mark.parametrize("impl", ["cuda", "triton"])
def test_build_quant_grid_mse_does_not_increase_error(impl):
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
        impl=impl,
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
        impl=impl,
    )
    scale_mse, qzero_mse = unpack_qmeta_tensor(qmeta_mse, group_size, R)
    _, y_mse = quantize_dequant(W, scale_mse, qzero_mse, maxq_mse)
    mse_mse = ((y_mse - W) ** 2).mean().item()

    # In practice mse_mse <= mse_abs; allow tiny numerical wiggle
    assert mse_mse <= mse_abs + 1e-5


@pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("impl", ["cuda", "triton"])
def test_build_quant_grid_supports_fp16_bf16(dtype, impl):
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
        impl=impl,
    )

    num_groups = (R + group_size - 1) // group_size
    assert qmeta.shape == (C, num_groups, 4)
    assert qmeta.dtype == torch.uint8
    assert maxq.shape == torch.Size([])


@pytest.mark.skipif(not (has_cuda() and has_fp8()), reason="float8 or CUDA not available")
@pytest.mark.parametrize("impl", ["cuda", "triton"])
def test_build_quant_grid_supports_fp8_e4m3(impl):
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
        impl=impl,
    )

    assert qmeta.shape == (C, 1, 4)
    assert maxq.item() == 2**4 - 1


# ---------- tests for solver ----------


@pytest.mark.parametrize("C,R", [(4, 32), (6, 64)])
@pytest.mark.parametrize("impl", ["cuda", "triton"])
def test_solver_matches_reference_cpu(C, R, impl):
    """
    Compare GPTQ.solver (CPU path) against a pure reference solver that
    decodes qmeta exactly like the CUDA implementation.
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

    # 1. Build qmeta on CPU
    qmeta, maxq, _ = gptq.build_quant_grid(
        W,
        group_size=group_size,
        bits=bits,
        symmetric=False,
        mode="absmax",
        impl=impl,
    )

    # 2. Reference solver (CPU, from qmeta)
    q_ref, W_ref = reference_gptq_solver_from_qmeta(
        W, Hinv, qmeta, group_size, bits
    )

    # 3. GPTQ.solver on CPU
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
def test_solver_cuda_matches_reference_cpu():
    """
    Compare GPTQ.solver CUDA path against the same reference CPU solver.
    """
    torch.manual_seed(0)
    C, R = 8, 256
    group_size = 128
    bits = 4

    # Base weights / Hessian on CPU
    W_cpu = torch.randn(C, R, dtype=torch.float32, device="cpu")
    X = torch.randn(C, C, dtype=torch.float32)
    H = X @ X.T + 1e-3 * torch.eye(C, dtype=torch.float32)
    Hinv_cpu = torch.inverse(H)

    gptq = GPTQ()

    # 1. Build qmeta on GPU (what the real pipeline will do)
    W_gpu = W_cpu.to("cuda")
    Hinv_gpu = Hinv_cpu.to("cuda")

    qmeta_gpu, maxq_gpu, _ = gptq.build_quant_grid(
        W_gpu,
        group_size=group_size,
        bits=bits,
        symmetric=False,
        mode="absmax",
        impl="cuda",
    )

    # 2. Reference solver from qmeta on CPU
    qmeta_cpu = qmeta_gpu.cpu()
    q_ref, W_ref = reference_gptq_solver_from_qmeta(
        W_cpu, Hinv_cpu, qmeta_cpu, group_size, bits
    )

    # 3. GPTQ.solver on CUDA
    W_solver_gpu = W_gpu.clone()
    q_gptq_gpu = gptq.solver(
        weight=W_solver_gpu,
        hessian_inv=Hinv_gpu,
        qmeta=qmeta_gpu,
        maxq=maxq_gpu,
        group_size=group_size,
        bits=bits,
    )

    # Move results back for comparison
    q_gptq = q_gptq_gpu.cpu()
    W_solver = W_solver_gpu.cpu()

    assert torch.equal(q_gptq, q_ref)
    assert torch.allclose(W_solver, W_ref, rtol=1e-5, atol=1e-6)
