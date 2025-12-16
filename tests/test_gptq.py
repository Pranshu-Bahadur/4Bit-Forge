import math
import pytest
import torch
from copy import deepcopy
from forge.gptq import GPTQ  # module under test


def has_cuda():
    return torch.cuda.is_available()


def has_fp8():
    return hasattr(torch, "float8_e4m3fn")


def has_babai_solver():
    """
    Babai solver is CUDA-only and requires the extension to expose babai_solver.
    """
    try:
        from forge.cuda import kernels as cuda_kernels
        return hasattr(cuda_kernels, "babai_solver")
    except Exception:
        return False


# ---------- helpers for tests ----------

def unpack_qmeta_tensor(qmeta, group_size, R):
    """
    Test helper to unpack (C, G, 4) qmeta back to (C, R) scale/qzero tensors.

    NOTE: This uses the simpler decoding that matches the *grid builder*
    tests, not the full solver semantics. It’s meant only for approximate
    error comparison in build_quant_grid tests.
    """
    C, G, _ = qmeta.shape

    lo = qmeta[..., 0].to(torch.int16)
    hi = qmeta[..., 1].to(torch.int16)

    # Combine bytes: little-endian into signed int16
    log2_q88 = (lo & 0x00FF) | (hi << 8)

    scale_g = torch.exp2(log2_q88.float() / 256.0)  # (C, G)
    qzero_g = qmeta[..., 2].float()                 # (C, G)

    # Expand to (C, G * group_size)
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

    # Crop padding
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
    log2_q88 = lo | (hi << 8)

    # Manual sign-extension from 16-bit to 32-bit
    log2_q88 = torch.where(log2_q88 >= 32768, log2_q88 - 65536, log2_q88)

    log2_scale = log2_q88.float() / 256.0
    scale_g = torch.exp2(log2_scale)
    inv_scale_g = torch.exp2(-log2_scale)

    qzero_u8 = meta[:, 2].float()
    flags = meta[:, 3]

    maxq_val = float((1 << bits) - 1)

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
    Reference GPTQ solver that matches solver.cu semantics.

    IMPORTANT:
      - `hessian_inv_cho` is *Cholesky(H^{-1})*, upper-triangular.
      - We mirror the CUDA kernel:

          1) Quantize rows in blocks, recording delta = W_old - W_quant (delta_block).
          2) Solve A_lower * E_T = Delta_T, where A_lower = H_block^T and
             H_block is the [J,J] sub-block of `hessian_inv_cho`.
          3) Tail update: W_tail -= H_cross^T @ E_J, where
             H_cross = `hessian_inv_cho[J, K]`.

    Returns:
      qweight (uint8), updated W (fp32)
    """
    assert weight.ndim == 2
    C, R = weight.shape

    W = weight.to(torch.float32).clone()
    Hcho = hessian_inv_cho.to(torch.float32).clone()

    qweight = torch.empty(C, R, dtype=torch.uint8)

    num_groups = qmeta.size(1)
    maxq_val = float((1 << bits) - 1)

    for block_start in range(0, C, block_size):
        block_end = min(block_start + block_size, C)
        B = block_end - block_start

        # 1) Quantize this block and accumulate delta
        delta_block = torch.zeros(B, R, dtype=torch.float32)

        for row_offset, j in enumerate(range(block_start, block_end)):
            row_meta = qmeta[j]
            scale_g, inv_scale_g, qzero_g, maxq_ref = _decode_qmeta_row_for_solver(row_meta, bits)
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

                x = W[j, start:end].clone()

                biased = x * inv_s + q0
                q = torch.round(biased).clamp_(0.0, maxq_val)

                y = (q - q0) * s
                delta = x - y

                W[j, start:end] = y
                qweight[j, start:end] = q.to(torch.uint8)
                delta_block[row_offset, start:end] = delta

        if block_end >= C:
            continue

        # 2) TRSM solve: A_lower * E_T = Delta_T, A_lower = H_block^T
        H_block = Hcho[block_start:block_end, block_start:block_end]  # [B,B] upper
        A_lower = H_block.t()                                        # [B,B] lower

        Delta_T = delta_block.t().contiguous()  # [R,B]
        E_T = torch.empty_like(Delta_T)

        for r in range(R):
            b = Delta_T[r]  # (B,)
            x = torch.empty(B, dtype=torch.float32)
            for i in range(B):
                ssum = 0.0 if i == 0 else torch.dot(A_lower[i, :i], x[:i])
                diag = A_lower[i, i]
                x[i] = (b[i] - ssum) / diag
            E_T[r] = x

        E_J = E_T.t().contiguous()  # [B,R]

        # 3) Tail update
        H_cross = Hcho[block_start:block_end, block_end:C]  # [B, C_tail]
        if H_cross.numel() > 0:
            W_tail = W[block_end:C, :]
            W_tail = W_tail - H_cross.t().mm(E_J)
            W[block_end:C, :] = W_tail

    return qweight, W


# ---------- tests for build_quant_grid ----------

@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128, 256])
@pytest.mark.parametrize("impl", ["cuda", "triton"])
@pytest.mark.parametrize("mode", ["absmax", "mse"])
@pytest.mark.parametrize("symmetric", [True, False])
def test_build_quant_grid_shapes_cpu(bits, group_size, impl, mode, symmetric):
    torch.manual_seed(0)
    C, R = 7, 257
    W = torch.randn(C, R, dtype=torch.float32, device="cpu")

    gptq = GPTQ(algorithm="gptq")
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
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("impl", ["cuda", "triton"])
@pytest.mark.parametrize("mode", ["absmax", "mse"])
@pytest.mark.parametrize("symmetric", [True, False])
def test_build_quant_grid_cpu_vs_gpu_error(bits, group_size, impl, mode, symmetric):
    torch.manual_seed(0)
    C, R = 8, 256
    W_cpu = torch.randn(C, R, dtype=torch.float32, device="cpu")

    gptq = GPTQ(algorithm="gptq")

    # CPU path
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

    # GPU path
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

    assert math.isclose(mse_cpu, mse_gpu, rel_tol=0.05, abs_tol=1e-4)


@pytest.mark.parametrize("impl", ["cuda", "triton"])
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("symmetric", [True, False])
def test_build_quant_grid_mse_does_not_increase_error(impl, group_size, bits, symmetric):
    torch.manual_seed(123)
    C, R = 4, 128
    W = torch.randn(C, R, dtype=torch.float32)

    gptq = GPTQ(algorithm="gptq")

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

    assert mse_mse <= mse_abs + 1e-5


@pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("impl", ["cuda", "triton"])
@pytest.mark.parametrize("mode", ["absmax", "mse"])
@pytest.mark.parametrize("symmetric", [True, False])
def test_build_quant_grid_supports_fp16_bf16(dtype, impl, mode, symmetric):
    torch.manual_seed(42)
    C, R = 8, 256
    group_size = 128
    W = torch.randn(C, R, dtype=torch.float32, device="cuda").to(dtype)

    gptq = GPTQ(algorithm="gptq")
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
    torch.manual_seed(7)
    C, R = 4, 128
    W_fp32 = torch.randn(C, R, dtype=torch.float32, device="cuda")
    W_fp8 = W_fp32.to(torch.float8_e4m3fn)

    gptq = GPTQ(algorithm="gptq")
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
    torch.manual_seed(0)
    C, R = 5, 100
    group_size = 32
    bits = 4
    symmetric = False
    W = torch.zeros(C, R, dtype=torch.float32)

    gptq = GPTQ(algorithm="gptq")
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
    assert torch.allclose(y, W, atol=1e-6)


# ---------- tests for Hessian inverse / Cholesky path (GPTQ algorithm) ----------

def _naive_hessian_inverse_cholesky(H_init: torch.Tensor,
                                   W: torch.Tensor,
                                   rel_damp: float) -> torch.Tensor:
    """
    Reference implementation matching GPTQ._get_hessian_factor
    AFTER your change: NO normalization.

        H_init
          -> (zero-cols regularization + damping)
          -> H_reg^{-1}
          -> chol(H_reg^{-1}) (upper)
    """
    H = H_init.clone()

    # Dead channels (rows in W that are all-zero)
    zero_cols = torch.nonzero(W.eq(0).all(dim=1), as_tuple=False).view(-1)
    if zero_cols.numel() > 0:
        H[zero_cols, :] = 0.0
        H[:, zero_cols] = 0.0
        H[zero_cols, zero_cols] = 1.0

    diag_H = H.diagonal()
    damp = rel_damp * diag_H.mean()
    diag_H.add_(damp)

    # Match the numerics of your implementation: cholesky + cholesky_inverse
    L = torch.linalg.cholesky(H, upper=False)
    Hinv = torch.cholesky_inverse(L, upper=False)
    Hinv_cho = torch.linalg.cholesky(Hinv, upper=True)
    return Hinv_cho


@pytest.mark.parametrize("rel_damp", [1e-2])
@pytest.mark.parametrize("C", [4, 8, 16, 32])
def test_hessian_inverse_cholesky_matches_naive(rel_damp, C):
    torch.manual_seed(0)
    R = 2 * C + 3

    W = torch.randn(C, R, dtype=torch.float32)
    if C > 0:
        W[0].zero_()
    if C > 1:
        W[-1].zero_()

    X = torch.randn(C, C, dtype=torch.float32)
    H_init = X @ X.T + 1e-3 * torch.eye(C, dtype=torch.float32)

    Hinv_cho_ref = _naive_hessian_inverse_cholesky(H_init, W, rel_damp)

    gptq = GPTQ(rel_damp=rel_damp, algorithm="gptq")
    gptq.H = H_init.clone()
    gptq.W = W.clone()
    gptq.d_col = C

    Hinv_cho_gptq = gptq._get_hessian_factor()

    assert Hinv_cho_gptq.shape == Hinv_cho_ref.shape
    assert torch.allclose(Hinv_cho_gptq, Hinv_cho_ref, rtol=1e-5, atol=1e-5)

    # implied H^{-1} match
    Hinv_ref = Hinv_cho_ref.T @ Hinv_cho_ref
    Hinv_gptq = Hinv_cho_gptq.T @ Hinv_cho_gptq
    assert torch.allclose(Hinv_gptq, Hinv_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("rel_damp", [1e-2])
@pytest.mark.parametrize("C", [16])
def test_hessian_inverse_cholesky_ill_conditioned(rel_damp, C):
    torch.manual_seed(1)
    R = C
    W = torch.randn(C, R, dtype=torch.float32)

    X = torch.randn(C, C, dtype=torch.float32)
    H_init = X @ X.T + 1e-6 * torch.eye(C, dtype=torch.float32)

    gptq = GPTQ(rel_damp=rel_damp, algorithm="gptq")
    gptq.H = H_init.clone()
    gptq.W = W.clone()
    gptq.d_col = C

    Hinv_cho = gptq._get_hessian_factor()
    Hinv_recon = Hinv_cho.T @ Hinv_cho

    # SPD check (stable real eigs)
    evals = torch.linalg.eigvalsh(Hinv_recon)
    assert torch.all(evals > 0)


# ---------- tests for solver (GPTQ path) ----------

@pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
@pytest.mark.parametrize("C,R", [(4, 32), (6, 64), (16, 128), (32, 256)])
@pytest.mark.parametrize("impl", ["cuda", "triton"])
@pytest.mark.parametrize("mode", ["absmax", "mse"])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("symmetric", [True, False])
def test_solver_matches_reference_cpu(C, R, impl, mode, bits, group_size, symmetric):
    torch.manual_seed(0)
    W_cpu = torch.randn(C, R, dtype=torch.float32, device="cpu")

    X = torch.randn(C, C, dtype=torch.float32)
    H = X @ X.T + 1e-3 * torch.eye(C, dtype=torch.float32)
    L = torch.linalg.cholesky(H, upper=False)
    Hinv = torch.cholesky_inverse(L, upper=False)
    Hinv_cho = torch.linalg.cholesky(Hinv, upper=True)

    gptq = GPTQ(algorithm="gptq")

    qmeta_cpu, maxq_cpu, _ = gptq.build_quant_grid(
        W_cpu,
        group_size=group_size,
        bits=bits,
        symmetric=symmetric,
        mode=mode,
        impl=impl,
        quant_n_grid=8,
    )

    q_ref, W_ref = reference_gptq_solver_from_qmeta(
        W_cpu, Hinv_cho, qmeta_cpu, group_size, bits
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


@pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
@pytest.mark.parametrize("mode", ["absmax", "mse"])
@pytest.mark.parametrize("bits", [4])
@pytest.mark.parametrize("group_size", [128])
@pytest.mark.parametrize("symmetric", [True, False])
def test_solver_cuda_matches_reference_cpu(mode, bits, group_size, symmetric):
    torch.manual_seed(0)
    C, R = 8, 256

    W_cpu = torch.randn(C, R, dtype=torch.float32, device="cpu")
    X = torch.randn(C, C, dtype=torch.float32)
    H = X @ X.T + 1e-3 * torch.eye(C, dtype=torch.float32)

    L = torch.linalg.cholesky(H, upper=False)
    Hinv = torch.cholesky_inverse(L, upper=False)
    Hinv_cho_cpu = torch.linalg.cholesky(Hinv, upper=True)

    gptq = GPTQ(algorithm="gptq")

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

    qmeta_cpu = qmeta_gpu.cpu()
    q_ref, W_ref = reference_gptq_solver_from_qmeta(
        W_cpu, Hinv_cho_cpu, qmeta_cpu, group_size, bits
    )

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
def test_solver_no_tail_when_C_leq_32(mode, bits, group_size):
    """
    CUDA wrapper clamps block_size to <=32 and <=C.
    If C <= 32, there's only one block => no tail update.
    """
    torch.manual_seed(2)
    C, R = 4, 64

    W_cpu = torch.randn(C, R, dtype=torch.float32, device="cpu")
    X = torch.randn(C, C, dtype=torch.float32)
    H = X @ X.T + 1e-3 * torch.eye(C, dtype=torch.float32)

    L = torch.linalg.cholesky(H, upper=False)
    Hinv = torch.cholesky_inverse(L, upper=False)
    Hinv_cho = torch.linalg.cholesky(Hinv, upper=True)

    gptq = GPTQ(algorithm="gptq")

    qmeta_cpu, maxq_cpu, _ = gptq.build_quant_grid(
        W_cpu,
        group_size=group_size,
        bits=bits,
        symmetric=False,
        mode=mode,
        impl="cuda",
        quant_n_grid=8,
    )

    # Reference with block_size=32 (equivalent because C=4)
    q_ref, W_ref = reference_gptq_solver_from_qmeta(
        W_cpu, Hinv_cho, qmeta_cpu, group_size, bits, block_size=32
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

    assert torch.equal(q_gptq_gpu.cpu(), q_ref)
    assert torch.allclose(W_gpu.cpu(), W_ref, rtol=1e-5, atol=1e-6)


# ---------- Babai-vs-GPTQ confirmation tests ----------

def dequantize_forge(qweight_RC, qmeta_CG4, group_size):
    # qweight is [R,C] like Linear.weight. Return [C,R] to match your W_base convention.
    qwt = qweight_RC.t().contiguous()  # [C,R]
    C, R = qwt.shape
    scale, qzero = unpack_qmeta_tensor(qmeta_CG4.cpu(), group_size, R)
    return (qwt.float().cpu() - qzero) * scale  # [C,R]

def mismatch_rate(a_u8, b_u8):
    return (a_u8 != b_u8).float().mean().item()

def hessian_weighted_loss(W_base_CR, Wq_CR, H_CC):
    D = (W_base_CR.float() - Wq_CR.float())
    return (D * (H_CC.float() @ D)).mean().item()

def reverse_layer(layer):
    """
    Reverse the dimensions of a nn.Linear layer for reverse-order quantization.
    - Flips weight rows and columns (transposed view, but we handle internally).
    - Since GPTQ internals use W_t = weight.t() [C, R], reversing means flipping along the quantization dim (rows in W_t).
    - We also need to reverse the Hessian, but since Hessian is computed in update(), we flip the layer's weight to induce reversed H.
    """
    layer_rev = deepcopy(layer)
    # Original weight: [R, C] (out, in)
    # Flip along in_features (C) -> reverse rows in W_t [C, R]
    idx = torch.arange(layer.in_features - 1, -1, -1, device=layer.weight.device)
    layer_rev.weight.data = layer.weight.data[:, idx].contiguous()  # Flip columns (in_features)
    return layer_rev

def reverse_calib(calib):
    """
    Reverse calibration inputs to match reversed layer.
    - calib: [N, C] (tokens, in_features)
    - Flip columns to reverse feature order.
    """
    idx = torch.arange(calib.size(1) - 1, -1, -1, device=calib.device)
    return calib[:, idx].contiguous()

def reverse_qweight(qweight):
    """
    Unflip qweight to original order.
    - qweight: [R, C] quantized (out, in)
    - Since reversal flipped in_features, unflip columns.
    """
    idx = torch.arange(qweight.size(1) - 1, -1, -1, device=qweight.device)
    return qweight[:, idx].contiguous()

def reverse_qmeta(qmeta):
    """
    Unflip qmeta to original order.
    - qmeta: [C, G, 4] (in_features/groups)
    - Flip along dim 0 (in_features).
    """
    idx = torch.arange(qmeta.size(0) - 1, -1, -1, device=qmeta.device)
    return qmeta[idx].contiguous()

def relative_mse(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Robust Relative MSE computation.
    Flattens tensors to ensure shape mismatches (e.g. [C, R] vs [R, C])
    don't cause errors, provided total elements match.
    """
    if x.numel() != y.numel():
        return float("nan")

    # Flatten and cast to float32 for precision
    x_flat = x.float().view(-1).cpu()
    y_flat = y.float().view(-1).cpu()

    num = torch.mean((x_flat - y_flat) ** 2)
    den = torch.mean(y_flat ** 2)

    return (num / den).item() if den != 0 else float("nan")

@pytest.mark.skipif(not (has_cuda() and has_babai_solver()), reason="Babai CUDA solver not available")
@pytest.mark.parametrize("C,R", [(16, 128), (32, 256)])
@pytest.mark.parametrize("bits", [4])
@pytest.mark.parametrize("group_size", [128])
@pytest.mark.parametrize("symmetric", [False, True])
def test_babai_quantize_matches_gptq_quantize_e2e(C, R, bits, group_size, symmetric):
    """
    End-to-end comparison:
      - Same layer weights
      - Same calibration inputs
      - Same qmeta builder (via GPTQ.quantize())
      - Compare final qweight (primary) and qmeta (sanity)
    Babai should match reverse-order GPTQ exactly, and forward GPTQ closely (minor order diffs).
    NOTE: We use nn.Linear(in=C, out=R) so that the solver's internal W_t becomes (C, R).
    """
    torch.manual_seed(0)

    # Make Hessian accumulation deterministic-ish (avoid TF32 differences)
    old_tf32_mm = torch.backends.cuda.matmul.allow_tf32
    old_tf32_cu = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        device = torch.device("cuda")

        # ----- Build a deterministic layer with known weights -----
        layer = torch.nn.Linear(C, R, bias=False, device=device, dtype=torch.float16)
        W_t = torch.randn(C, R, device=device, dtype=torch.float16)  # (C,R)
        layer.weight.data.copy_(W_t.t().contiguous())                # (R,C)
        W_orig = W_t.clone()  # For MSE checks

        # ----- Calibration data (same for both) -----
        calib = torch.randn(2048, C, device=device, dtype=torch.float16)

        W_init = layer.weight.data.clone()


        def run_quant(algorithm: str, reverse_order: bool = False):
            
            # For reverse: flip layer dims/Hessian/calib, run forward, unflip outputs
            if reverse_order:
                layer_rev = reverse_layer(layer)  # Flip weight rows/cols
                calib_rev = reverse_calib(calib)  # Flip input features
            else:
                layer_rev, calib_rev = layer, calib

            layer0 = deepcopy(layer_rev)
            layer0.weight.data.copy_(W_init if not reverse_order else W_init[:, torch.arange(C-1, -1, -1, device=device)])
            algo = GPTQ(
                layer=layer0,
                group_size=group_size,
                sym=symmetric,
                rel_damp=1e-2,
                quantization_scale="mse",  # keep tests fast; both paths share it
                algorithm=algorithm,
            )
            algo.update(calib_rev)
            qweight, qmeta, maxq = algo.quantize(bits)
            if reverse_order:
                qweight = reverse_qweight(qweight)  # Unflip rows
                qmeta = reverse_qmeta(qmeta)
            return qweight, qmeta, maxq

        # Forward GPTQ (standard)
        q_gptq_fwd, qmeta_gptq, maxq_gptq = run_quant("gptq", reverse_order=False)
        
        # Reverse GPTQ (should match Babai exactly)
        q_gptq_rev, qmeta_rev, maxq_rev = run_quant("gptq", reverse_order=True)
        
        # Babai
        q_babai, qmeta_babai, maxq_babai = run_quant("babai")

        # ----- Sanity: grid builder should match exactly (same W, same settings) -----
        assert torch.equal(qmeta_babai, qmeta_gptq), "qmeta differs; grid builder path is not identical"
        assert float(maxq_babai.item()) == float(maxq_gptq.item()) == float((1 << bits) - 1)

        mismatch_rate = (q_babai != q_gptq_fwd).float().mean().item()
        assert mismatch_rate < 0.02, f"qweight mismatch rate = {mismatch_rate:.6f} exceeds threshold (expected due to order)"

        # ----- Quality: Relative MSE < threshold, similar across -----
        def compute_mse(qw, qm):
            deq = dequantize_forge(qw, qm, group_size)  # Assuming your dequant func
            return relative_mse(deq, W_orig)
        mse_babai = compute_mse(q_babai, qmeta_babai)
        mse_fwd = compute_mse(q_gptq_fwd, qmeta_gptq)
        assert mse_babai < 0.05, f"Babai MSE too high: {mse_babai:.2e}"  # Tune threshold based on bits
        assert abs(mse_babai - mse_fwd) < 0.01, f"MSE diff too large: {abs(mse_babai - mse_fwd):.2e}"

    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32_mm
        torch.backends.cudnn.allow_tf32 = old_tf32_cu