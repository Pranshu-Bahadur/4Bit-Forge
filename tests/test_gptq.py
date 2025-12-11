import math
import pytest
import torch

from forge.gptq import GPTQ  # Assuming this is the module under test

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

    # Symmetric override: qzero = (maxq + 1)/2
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

@pytest.mark.parametrize("bits", [2, 3, 4, 8])  # Expanded bits
@pytest.mark.parametrize("group_size", [32, 64, 128, 256])  # More group sizes
@pytest.mark.parametrize("impl", ["cuda", "triton"])
@pytest.mark.parametrize("mode", ["absmax", "mse"])
@pytest.mark.parametrize("symmetric", [True, False])  # Added symmetric
def test_build_quant_grid_shapes_cpu(bits, group_size, impl, mode, symmetric):
    torch.manual_seed(0)
    C, R = 7, 257  # odd dims to test padding logic
    W = torch.randn(C, R, dtype=torch.float32, device="cpu")

    gptq = GPTQ()
    # keep quant_n_grid small in tests when using mse
    qmeta, maxq, pad = gptq.build_quant_grid(
        W,
        group_size=group_size,
        bits=bits,
        symmetric=symmetric,
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
@pytest.mark.parametrize("bits", [2, 3, 4, 8])  # Expanded
@pytest.mark.parametrize("group_size", [32, 128])  
@pytest.mark.parametrize("impl", ["cuda", "triton"])
@pytest.mark.parametrize("mode", ["absmax", "mse"])
@pytest.mark.parametrize("symmetric", [True, False])  # Added
def test_build_quant_grid_cpu_vs_gpu_error(bits, group_size, impl, mode, symmetric):
    """
    Compare CPU vs GPU grid builder indirectly by comparing quantization error.

    Note: GPU uses packed Q8.8 scales, CPU uses pure float32 emulation.
    Small precision diffs are expected due to Q8.8 encoding.
    """
    torch.manual_seed(0)
    C, R = 8, 256
    W_cpu = torch.randn(C, R, dtype=torch.float32, device="cpu")

    gptq = GPTQ()

    # 1. CPU path
    qmeta_cpu, maxq_cpu, _ = gptq.build_quant_grid(
        W_cpu,
        group_size=group_size,
        bits=bits,
        symmetric=symmetric,
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
        symmetric=symmetric,
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
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("symmetric", [True, False])
def test_build_quant_grid_mse_does_not_increase_error(impl, group_size, bits, symmetric):
    """
    Check that MSE mode doesn't worsen error compared to absmax.
    """
    torch.manual_seed(123)
    C, R = 4, 128
    W = torch.randn(C, R, dtype=torch.float32)

    gptq = GPTQ()

    # Baseline absmax
    qmeta_abs, maxq_abs, _ = gptq.build_quant_grid(
        W,
        group_size=group_size,
        bits=bits,
        symmetric=symmetric,
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
        bits=bits,
        symmetric=symmetric,
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
@pytest.mark.parametrize("symmetric", [True, False])
def test_build_quant_grid_supports_fp16_bf16(dtype, impl, mode, symmetric):
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
        symmetric=symmetric,
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
@pytest.mark.parametrize("symmetric", [True, False])
def test_build_quant_grid_supports_fp8_e4m3(impl, mode, symmetric):
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
        symmetric=symmetric,
        mode=mode,
        impl=impl,
        quant_n_grid=8,
    )

    assert qmeta.shape == (C, 1, 4)
    assert maxq.item() == 2**4 - 1

@pytest.mark.parametrize("impl", ["cuda", "triton"])
@pytest.mark.parametrize("mode", ["absmax", "mse"])
def test_build_quant_grid_zero_weights(impl, mode):
    """
    Test with all-zero weights to check padding and zero handling.
    """
    torch.manual_seed(0)
    C, R = 5, 100
    group_size = 32
    bits = 4
    symmetric = False
    W = torch.zeros(C, R, dtype=torch.float32)

    gptq = GPTQ()
    qmeta, maxq, pad = gptq.build_quant_grid(
        W,
        group_size=group_size,
        bits=bits,
        symmetric=symmetric,
        mode=mode,
        impl=impl,
        quant_n_grid=8,
    )

    num_groups = (R + group_size - 1) // group_size
    assert qmeta.shape == (C, num_groups, 4)
    assert pad == (num_groups * group_size) - R

    scale, qzero = unpack_qmeta_tensor(qmeta, group_size, R)
    _, y = quantize_dequant(W, scale, qzero, maxq)
    assert torch.allclose(y, W, atol=1e-6)  # Should quantize to zero perfectly

# ---------- tests for Hessian inverse / Cholesky path ----------

def _naive_hessian_inverse_cholesky(H_init: torch.Tensor,
                                    W: torch.Tensor,
                                    rel_damp: float) -> torch.Tensor:
    """
    Reference implementation of the path:

        H_init
          -> (zero-cols regularization + damping)
          -> H_reg^{-1}
          -> chol(H_reg^{-1}) (upper)

    Matching GPTQ._get_hessian_inverse_cholesky's regularization logic.
    """
    H = H_init.clone()
    C = H.shape[0]

    # Identify dead input channels (rows in W that are all-zero)
    zero_cols = torch.nonzero(W.eq(0).all(dim=1), as_tuple=False).view(-1)
    if zero_cols.numel() > 0:
        H[zero_cols, :] = 0.0
        H[:, zero_cols] = 0.0
        H[zero_cols, zero_cols] = 1.0

    # Damping
    diag_H = H.diagonal()
    damp = rel_damp * diag_H.mean()
    diag_H.add_(damp)

    # Invert and take Cholesky
    Hinv = torch.inverse(H)
    Hinv_cho = torch.linalg.cholesky(Hinv, upper=True)  # [C, C]

    # Row-normalize as in GPTQ._get_hessian_inverse_cholesky
    diag_u = Hinv_cho.diagonal()
    diag_u = torch.where(diag_u == 0, torch.ones_like(diag_u), diag_u)
    Hinv_cho = Hinv_cho / diag_u.unsqueeze(-1)

    return Hinv_cho

@pytest.mark.parametrize("rel_damp", [1e-4, 1e-3, 1e-2, 1e-1])  # Expanded damping
@pytest.mark.parametrize("C", [4, 8, 16, 32])  # Larger C
def test_hessian_inverse_cholesky_matches_naive(rel_damp, C):
    """
    Compare GPTQ._get_hessian_inverse_cholesky against a naive path:

        H_init -> H_reg^{-1} -> chol(H_reg^{-1})

    under the *same* regularization and damping as the GPTQ implementation.
    """
    torch.manual_seed(0)
    R = 2 * C + 3

    # Random working weight (C, R) with some explicit dead rows to exercise logic
    W = torch.randn(C, R, dtype=torch.float32)
    if C > 0:
        W[0].zero_()   # guaranteed dead channel
    if C > 1:
        W[-1].zero_()  # another dead channel

    # SPD-ish Hessian
    X = torch.randn(C, C, dtype=torch.float32)
    H_init = X @ X.T + 1e-3 * torch.eye(C, dtype=torch.float32)

    # Naive reference
    Hinv_cho_ref = _naive_hessian_inverse_cholesky(H_init, W, rel_damp)

    # GPTQ path
    gptq = GPTQ(rel_damp=rel_damp)
    gptq.H = H_init.clone()
    gptq.W = W.clone()
    gptq.d_col = C

    Hinv_cho_gptq = gptq._get_hessian_inverse_cholesky()

    # Should be numerically very close
    assert Hinv_cho_gptq.shape == Hinv_cho_ref.shape
    assert torch.allclose(Hinv_cho_gptq, Hinv_cho_ref, rtol=1e-5, atol=1e-6)

    # Also check that the implied H^{-1} matrices match
    Hinv_ref = Hinv_cho_ref.T @ Hinv_cho_ref
    Hinv_gptq = Hinv_cho_gptq.T @ Hinv_cho_gptq
    assert torch.allclose(Hinv_gptq, Hinv_ref, rtol=1e-5, atol=1e-6)

@pytest.mark.parametrize("rel_damp", [1e-2])
@pytest.mark.parametrize("C", [16])
def test_hessian_inverse_cholesky_ill_conditioned(rel_damp, C):
    """
    Test with near-singular Hessian to check damping effect.
    """
    torch.manual_seed(1)
    R = C
    W = torch.randn(C, R, dtype=torch.float32)

    # Ill-conditioned Hessian
    X = torch.randn(C, C, dtype=torch.float32)
    H_init = X @ X.T + 1e-6 * torch.eye(C, dtype=torch.float32)  # Small diagonal

    gptq = GPTQ(rel_damp=rel_damp)
    gptq.H = H_init.clone()
    gptq.W = W.clone()
    gptq.d_col = C

    Hinv_cho = gptq._get_hessian_inverse_cholesky()
    Hinv_recon = Hinv_cho.T @ Hinv_cho

    # Check positive definite
    assert torch.all(torch.linalg.eigvals(Hinv_recon) > 0)

# ---------- tests for solver ----------

@pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
@pytest.mark.parametrize("C,R", [(4, 32), (6, 64), (16, 128), (32, 256)])  # Larger sizes
@pytest.mark.parametrize("impl", ["cuda", "triton"])
@pytest.mark.parametrize("mode", ["absmax", "mse"])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("symmetric", [True, False])
def test_solver_matches_reference_cpu(C, R, impl, mode, bits, group_size, symmetric):
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

    # 1) Build qmeta on CPU
    qmeta_cpu, maxq_cpu, _ = gptq.build_quant_grid(
        W_cpu,
        group_size=group_size,
        bits=bits,
        symmetric=symmetric,
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
@pytest.mark.parametrize("bits", [4])
@pytest.mark.parametrize("group_size", [128])
@pytest.mark.parametrize("symmetric", [True, False])
def test_solver_cuda_matches_reference_cpu(mode, bits, group_size, symmetric):
    """
    Larger-shape regression test: C=8, R=256.
    Again, compare CUDA solver against the CPU reference using Hinv_cho.
    """
    torch.manual_seed(0)
    C, R = 8, 256

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
        symmetric=symmetric,
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

@pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
@pytest.mark.parametrize("mode", ["absmax"])
@pytest.mark.parametrize("bits", [4])
@pytest.mark.parametrize("group_size", [32])
def test_solver_edge_case_block_size(mode, bits, group_size):
    """
    Test with block_size larger than C to check no-tail logic.
    """
    torch.manual_seed(2)
    C, R = 4, 64
    block_size = 8  # Larger than C

    W_cpu = torch.randn(C, R, dtype=torch.float32, device="cpu")
    X = torch.randn(C, C, dtype=torch.float32)
    H = X @ X.T + 1e-3 * torch.eye(C, dtype=torch.float32)
    Hinv = torch.inverse(H)
    Hinv_cho = torch.linalg.cholesky(Hinv, upper=True)

    gptq = GPTQ()

    qmeta_cpu, maxq_cpu, _ = gptq.build_quant_grid(
        W_cpu,
        group_size=group_size,
        bits=bits,
        symmetric=False,
        mode=mode,
        impl="cuda",
        quant_n_grid=8,
    )

    q_ref, W_ref = reference_gptq_solver_from_qmeta(
        W_cpu, Hinv_cho, qmeta_cpu, group_size, bits, block_size=block_size
    )

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

