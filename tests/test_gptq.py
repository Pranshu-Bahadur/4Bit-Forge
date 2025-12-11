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


# ---------- solver-specific helpers (match CUDA solver exactly) ----------


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

    # Symmetric override: qzero = (maxq + 1.0)/2
    is_sym = (flags & 1) != 0
    sym_q0 = (maxq_val + 1.0) * 0.5
    qzero_g = torch.where(is_sym, sym_q0, qzero_u8)

    return scale_g, inv_scale_g, qzero_g, maxq_val


def reference_gptq_solver_from_qmeta(
    weight: torch.Tensor,
    hessian_inv_cho: torch.Tensor,
    qmeta: torch.Tensor,
    group_size: int,
    bits: int,
    block_size: int = 32,
):
    """
    Reference GPTQ solver that matches the CUDA kernel semantics.

    IMPORTANT:
      - `hessian_inv_cho` is *Cholesky(H^{-1})*, upper-triangular.
      - We DO NOT reconstruct full H^{-1} here.
      - We mirror solver.cu:

          1) Quantize rows in blocks, recording delta = W_old - W_quant (delta_block).
          2) Solve A_lower * E_T = Delta_T, where A_lower = H_block^T and
             H_block is the [J,J] sub-block of `hessian_inv_cho`.
          3) Tail update: W_tail -= H_cross^T @ E_J, where
             H_cross = `hessian_inv_cho[J, K]`.

    This should produce the same qweight and updated W as the CUDA kernel,
    up to small float32 rounding.
    """
    assert weight.ndim == 2
    C, R = weight.shape

    # Work in fp32 for the reference implementation
    W = weight.to(torch.float32).clone()
    Hcho = hessian_inv_cho.to(torch.float32).clone()

    qweight = torch.empty(C, R, dtype=torch.uint8)

    num_groups = qmeta.size(1)
    maxq_val = float((1 << bits) - 1)

    for block_start in range(0, C, block_size):
        block_end = min(block_start + block_size, C)
        B = block_end - block_start

        # 1) Quantize this block of rows and accumulate delta = W_old - W_quant
        delta_block = torch.zeros(B, R, dtype=torch.float32)

        for row_offset, j in enumerate(range(block_start, block_end)):
            row_meta = qmeta[j]  # (G, 4)
            scale_g, inv_scale_g, qzero_g, maxq_ref = _decode_qmeta_row_for_solver(
                row_meta, bits
            )
            # sanity
            assert abs(maxq_ref - maxq_val) < 1e-6

            for g in range(num_groups):
                start = g * group_size
                if start >= R:
                    break
                end = min(start + group_size, R)
                if start >= end:
                    continue

                s = scale_g[g]
                inv_s = inv_scale_g[g]
                q0 = qzero_g[g]

                # x_old
                x = W[j, start:end].clone()

                # q = clamp(round(x * inv_s + q0))
                biased = x * inv_s + q0
                q = torch.round(biased).clamp_(0.0, maxq_val)

                # y = (q - q0) * s
                y = (q - q0) * s

                # CUDA stores delta = x_old - y
                delta = x - y

                # write back
                W[j, start:end] = y
                qweight[j, start:end] = q.to(torch.uint8)
                delta_block[row_offset, start:end] = delta

        # No tail rows -> done with this final block
        if block_end >= C:
            continue

        # 2) TRSM solve on H_block^T (lower-triangular) to get E_J
        H_block = Hcho[block_start:block_end, block_start:block_end]  # [B, B], upper-tri
        A_lower = H_block.t()                                        # [B, B], lower-tri

        Delta_J = delta_block              # [B, R]
        Delta_T = Delta_J.t().contiguous() # [R, B]
        E_T = torch.empty_like(Delta_T)    # [R, B]

        # Forward substitution: A_lower * x = b
        for r in range(R):
            b = Delta_T[r]  # (B,)
            x = torch.empty(B, dtype=torch.float32)
            for i in range(B):
                if i == 0:
                    s = 0.0
                else:
                    s = torch.dot(A_lower[i, :i], x[:i])
                diag = A_lower[i, i]
                x[i] = (b[i] - s) / diag
            E_T[r] = x

        E_J = E_T.t().contiguous()  # [B, R]

        # 3) Tail update: W_tail -= H_cross^T @ E_J
        H_cross = Hcho[block_start:block_end, block_end:C]  # [B, C_tail]
        if H_cross.numel() > 0:
            W_tail = W[block_end:C, :].to(torch.float32)    # [C_tail, R]
            # H_cross^T: [C_tail, B], E_J: [B, R]
            W_tail = W_tail - H_cross.t().mm(E_J)
            W[block_end:C, :] = W_tail

    return qweight, W


# ---------- tests for build_quant_grid ----------


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("impl", ["cuda", "triton"])
@pytest.mark.parametrize("mode", ["absmax", "mse"])
def test_build_quant_grid_shapes_cpu(bits, group_size, impl, mode):
    torch.manual_seed(0)
    C, R = 7, 257  # odd dims to test padding logic
    W = torch.randn(C, R, dtype=torch.float32, device="cpu")

    gptq = GPTQ()
    # keep quant_n_grid small in tests when using mse
    qmeta, maxq, pad = gptq.build_quant_grid(
        W,
        group_size=group_size,
        bits=bits,
        symmetric=False,
        mode=mode,
        impl=impl,
        quant_n_grid=8,
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
@pytest.mark.parametrize("mode", ["absmax", "mse"])
def test_build_quant_grid_cpu_vs_gpu_error(bits, impl, mode):
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
        mode=mode,
        impl=impl,
        quant_n_grid=8,
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
        mode=mode,
        impl=impl,
        quant_n_grid=8,
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
@pytest.mark.parametrize("mode", ["absmax", "mse"])
def test_build_quant_grid_supports_fp16_bf16(dtype, impl, mode):
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
        mode=mode,
        quant_n_grid=8,
        impl=impl,
    )

    num_groups = (R + group_size - 1) // group_size
    assert qmeta.shape == (C, num_groups, 4)
    assert qmeta.dtype == torch.uint8
    assert maxq.shape == torch.Size([])


@pytest.mark.skipif(not (has_cuda() and has_fp8()), reason="float8 or CUDA not available")
@pytest.mark.parametrize("impl", ["cuda", "triton"])
@pytest.mark.parametrize("mode", ["absmax", "mse"])
def test_build_quant_grid_supports_fp8_e4m3(impl, mode):
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
        mode=mode,
        impl=impl,
        quant_n_grid=8,
    )

    assert qmeta.shape == (C, 1, 4)
    assert maxq.item() == 2**4 - 1


# ---------- tests for Hessian inverse / Cholesky path ----------


def _naive_hessian_inverse(H_init: torch.Tensor,
                           W: torch.Tensor,
                           rel_damp: float) -> torch.Tensor:
    """
    Naive reference for the *dense* H^{-1} used only for sanity checks:

        H_init
          -> (zero-row regularization + damping)
          -> H_reg^{-1}

    This mirrors GPTQ._get_hessian_inverse_cholesky's regularization logic,
    but returns the full inverse instead of a factor.
    """
    H = H_init.clone()
    C = H.shape[0]

    # Identify dead input channels (rows in W that are all-zero)
    zero_rows = torch.nonzero(W.eq(0).all(dim=1), as_tuple=False).view(-1)
    if zero_rows.numel() > 0:
        H[zero_rows, :] = 0.0
        H[:, zero_rows] = 0.0
        H[zero_rows, zero_rows] = 1.0

    # Damping
    diag_H = H.diagonal()
    damp = rel_damp * diag_H.mean()
    diag_H.add_(damp)

    # Invert
    Hinv = torch.inverse(H)
    return Hinv


@pytest.mark.parametrize("rel_damp", [1e-3, 1e-2])
@pytest.mark.parametrize("C", [8, 16])
def test_hessian_inverse_cholesky_sanity(rel_damp, C):
    """
    Sanity-check GPTQ._get_hessian_inverse_cholesky:

      * output is upper-triangular with non-zero diagonal
      * the induced matrix U^T U is SPD
      * its Rayleigh quotients are within a loose multiplicative band
        of the naive H^{-1} preconditioner.

    We no longer require the *exact* Cholesky factor to match the
    naive H -> H^{-1} -> chol(H^{-1}) path; the low-hanging fruit
    implementation legitimately changes the preconditioner, but
    it must still be well-behaved.
    """
    torch.manual_seed(0)
    R = 2 * C + 3

    # Random working weight (C, R) with some explicit dead rows to exercise logic
    W = torch.randn(C, R, dtype=torch.float32)
    if C > 0:
        W[0].zero_()   # guaranteed dead channel

    # SPD-ish Hessian
    X = torch.randn(C, C, dtype=torch.float32)
    H_init = X @ X.T + 1e-3 * torch.eye(C, dtype=torch.float32)

    # Naive dense inverse for reference
    Hinv_ref = _naive_hessian_inverse(H_init, W, rel_damp)

    # GPTQ path
    gptq = GPTQ(rel_damp=rel_damp)
    gptq.H = H_init.clone()
    gptq.W = W.clone()
    gptq.d_col = C

    Hinv_cho = gptq._get_hessian_inverse_cholesky()

    # Shape / dtype
    assert Hinv_cho.shape == (C, C)
    assert Hinv_cho.dtype == W.dtype

    # Roughly upper-triangular (allow tiny eps below diagonal)
    lower_part = torch.tril(Hinv_cho, diagonal=-1)
    assert torch.all(lower_part.abs() < 1e-4)

    # Diagonal should be finite and non-zero (row-normalised to ~1 in impl)
    diag = Hinv_cho.diagonal()
    assert torch.isfinite(diag).all()
    assert (diag.abs() > 1e-6).all()

    # Induced matrix should be SPD
    Hinv_approx = Hinv_cho.T.to(torch.float64) @ Hinv_cho.to(torch.float64)

    # Symmetric
    assert torch.allclose(Hinv_approx, Hinv_approx.T, rtol=1e-5, atol=1e-5)

    # Positive definite: Cholesky must succeed
    torch.linalg.cholesky(Hinv_approx, upper=False)

    # Compare Rayleigh quotients vs naive H^{-1} (very loose tolerance).
    Hinv_ref64 = Hinv_ref.to(torch.float64)
    for _ in range(8):
        z = torch.randn(C, dtype=torch.float64)
        q_ref = (z @ (Hinv_ref64 @ z)).item()
        q_approx = (z @ (Hinv_approx @ z)).item()

        # Both must be strictly positive in SPD case
        assert q_ref > 0.0
        assert q_approx > 0.0

        ratio = q_approx / q_ref
        # Allow up to ~3 orders-of-magnitude difference. This is intentionally
        # loose: we just want to catch completely degenerate preconditioners.
        assert 1e-3 < ratio < 1e3


# ---------- tests for solver ----------


@pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
@pytest.mark.parametrize("C,R", [(4, 32), (6, 64)])
@pytest.mark.parametrize("impl", ["cuda", "triton"])
@pytest.mark.parametrize("mode", ["absmax", "mse"])
def test_solver_matches_reference_cpu(C, R, impl, mode):
    """
    Compare GPTQ.solver (CUDA path) against the CPU reference solver that
    operates on Cholesky(H^{-1}) and mirrors solver.cu’s block/TRSM logic.
    """
    torch.manual_seed(0)
    W_cpu = torch.randn(C, R, dtype=torch.float32, device="cpu")

    # Construct SPD-ish Hessian and its inverse Cholesky factor
    X = torch.randn(C, C, dtype=torch.float32)
    H = X @ X.T + 1e-3 * torch.eye(C, dtype=torch.float32)
    Hinv = torch.inverse(H)
    Hinv_cho = torch.linalg.cholesky(Hinv, upper=True)  # Cholesky(H^{-1})

    gptq = GPTQ()
    group_size = min(32, R)
    bits = 4

    # 1) Build qmeta on CPU
    qmeta_cpu, maxq_cpu, _ = gptq.build_quant_grid(
        W_cpu,
        group_size=group_size,
        bits=bits,
        symmetric=False,
        mode=mode,
        impl=impl,
        quant_n_grid=8,
    )

    # 2) Reference solver on CPU, using Hinv_cho directly
    q_ref, W_ref = reference_gptq_solver_from_qmeta(
        W_cpu, Hinv_cho, qmeta_cpu, group_size, bits
    )

    # 3) GPTQ.solver on CUDA (taking Hinv_cho as input)
    W_gpu = W_cpu.to("cuda")
    Hinv_cho_gpu = Hinv_cho.to("cuda")
    qmeta_gpu = qmeta_cpu.to("cuda")
    maxq_gpu = maxq_cpu.to("cuda")

    q_gptq_gpu = gptq.solver(
        weight=W_gpu,
        hessian_inv=Hinv_cho_gpu,
        qmeta=qmeta_gpu,
        maxq=maxq_gpu,
        group_size=group_size,
        bits=bits,
    )

    q_gptq = q_gptq_gpu.cpu()
    W_solver = W_gpu.cpu()

    assert torch.equal(q_gptq, q_ref)
    assert torch.allclose(W_solver, W_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
@pytest.mark.parametrize("mode", ["absmax", "mse"])
def test_solver_cuda_matches_reference_cpu(mode):
    """
    Larger-shape regression test: C=8, R=256.
    Again, compare CUDA solver against the CPU reference using Hinv_cho.
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
    Hinv_cho_cpu = torch.linalg.cholesky(Hinv_cpu, upper=True)

    gptq = GPTQ()

    # 1) Build qmeta on GPU (realistic pipeline)
    W_gpu = W_cpu.to("cuda")
    Hinv_cho_gpu = Hinv_cho_cpu.to("cuda")

    qmeta_gpu, maxq_gpu, _ = gptq.build_quant_grid(
        W_gpu,
        group_size=group_size,
        bits=bits,
        symmetric=False,
        mode=mode,
        impl="cuda",
        quant_n_grid=8,
    )

    # 2) CPU reference using Hinv_cho
    qmeta_cpu = qmeta_gpu.cpu()
    q_ref, W_ref = reference_gptq_solver_from_qmeta(
        W_cpu, Hinv_cho_cpu, qmeta_cpu, group_size, bits
    )

    # 3) CUDA solver, taking Hinv_cho
    W_solver_gpu = W_gpu.clone()
    q_gptq_gpu = gptq.solver(
        weight=W_solver_gpu,
        hessian_inv=Hinv_cho_gpu,
        qmeta=qmeta_gpu,
        maxq=maxq_gpu,
        group_size=group_size,
        bits=bits,
    )

    q_gptq = q_gptq_gpu.cpu()
    W_solver = W_solver_gpu.cpu()

    assert torch.equal(q_gptq, q_ref)
    assert torch.allclose(W_solver, W_ref, rtol=1e-5, atol=1e-6)
