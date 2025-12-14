import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.conv import _ConvNd

from forge.cuda import kernels as cuda_kernels
from forge.trt import kernels as triton_kernels


class GPTQ:
    """
    GPTQ grid builder + solver (qmeta4-based) + optional MoE-Quant style
    per-layer Hessian accumulation and quantization pipeline.

    Usage modes
    -----------
    1) Stateless:
       - Use `build_quant_grid(weight, ...)` and `solver(weight, hessian_inv_cholesky, ...)`
         directly with any (C, R) transposed weight matrix.

    2) Layer-bound (MoE-Quant style):
       - Construct with a `layer` (nn.Linear / ConvNd) and hyperparameters.
       - Call `update(input)` repeatedly on calibration batches to accumulate Hessian.
       - Call `quantize(bits)` once to run GPTQ using 4Bit-Forge's solver, returning:
           qweight: (out_features, in_features) uint8
           qmeta:   (in_features, num_groups, 4) uint8
           maxq:    scalar tensor (2**bits - 1)

    Notes
    -----
    - Hessian is over the *input* dimension (d_col).
    - Solver expects `hessian_inv` to be the **Cholesky factor of H^{-1}**:
        H^{-1} = U^T U, where `hessian_inv` == U (upper-triangular).
    """

    # ------------------------------------------------------------------ #
    # Constructor / basic state
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        device: str | torch.device | None = None,
        layer: nn.Module | None = None,
        group_size: int | None = None,
        sym: bool = False,
        rel_damp: float = 1e-2,
        quantization_scale: str = "absmax",
        algorithm : str = "babai",
    ):
        # Optional device hint (mostly for stateless usage)
        if device is None or device == "":
            self.device = None
        else:
            self.device = torch.device(device)
        
        # inside __init__(...)
        self.algorithm = str(algorithm).lower()
        if self.algorithm not in ("gptq", "babai"):
            raise ValueError(f"Unknown algorithm={algorithm}. Expected 'gptq' or 'babai'.")


        # Optional bound layer (MoE-Quant style)
        self.layer: nn.Module | None = layer

        # Quantization hyperparameters
        self.group_size = group_size
        self.sym = sym
        self.rel_damp = rel_damp
        self.quantization_scale = quantization_scale  # "absmax" or "mse"

        # Hessian & calibration state
        self.H: torch.Tensor | None = None
        self.num_samples: int = 0  # tokens collected

        # Layer / weight metadata (only used when layer is provided)
        self.W: torch.Tensor | None = None
        self.d_row: int | None = None  # out_features
        self.d_col: int | None = None  # in_features (Hessian dimension == C)
        self.W_device: torch.device | None = None
        self.W_dtype: torch.dtype | None = None
        self.W_shape: torch.Size | None = None

        # Issue flags
        self.issue_zero_samples: bool = False
        self.issue_nan_hessian: bool = False
        self.issue_non_invertible: bool = False

        if layer is not None:
            self._init_from_layer(layer)

    # ------------------------------------------------------------------ #
    # Internal: initialize metadata from a bound layer
    # ------------------------------------------------------------------ #
    def _init_from_layer(self, layer: nn.Module) -> None:
        if not hasattr(layer, "weight"):
            raise TypeError("GPTQ layer must have a .weight attribute (e.g., nn.Linear or Conv).")

        W = layer.weight
        self.W_device = W.device
        self.W_dtype = W.dtype
        self.W_shape = W.shape

        if isinstance(layer, _ConvNd):
            # For convs, Hessian dimension is flattened input kernel dimension
            W_flat = W.flatten(1, -1)
            self.d_row = W_flat.shape[0]  # out_channels
            self.d_col = W_flat.shape[1]  # in_channels * kH * kW
        else:
            # Linear or other 2D weight: (out_features, in_features)
            assert W.ndim == 2, "Expected weight to be 2D for non-conv layers."
            self.d_row, self.d_col = W.shape

        # Store raw weight reference; we create working copies later
        self.W = W

    # ------------------------------------------------------------------ #
    # MoE-Quant–style Hessian accumulation
    # ------------------------------------------------------------------ #
    @property
    def tokens_collected(self) -> int:
        """Number of calibration tokens used to build the Hessian."""
        return self.num_samples

    @torch.no_grad()
    def update(self, input: torch.Tensor) -> None:
        """
        Update the estimate of the Hessian matrix from a batch of layer inputs.

        Args
        ----
        input: batch of layer inputs (same shape as passed to the layer.forward)
        """
        if self.layer is None:
            raise RuntimeError("GPTQ.update() requires a bound layer (pass layer=... in __init__).")
        if self.d_col is None:
            self._init_from_layer(self.layer)

        # Initialize Hessian if needed
        if self.H is None:
            self.H = torch.zeros(
                (self.d_col, self.d_col),
                device=input.device,
                dtype=torch.float32,  # keep Hessian in fp32 for stability
            )

        # Reshape input to (num_tokens, d_col)
        if isinstance(self.layer, nn.Linear):
            inp = input.reshape(-1, input.shape[-1])
        elif isinstance(self.layer, _ConvNd):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            # (N, C * prod(kernel_size), L) -> (N*L, C * prod(kernel_size))
            inp = unfold(input).transpose(1, 2).flatten(0, 1)
        else:
            raise TypeError("GPTQ.update() currently supports nn.Linear and ConvNd layers only.")

        inp = inp.float()
        num_new_samples = inp.shape[0]

        # Streaming Hessian update:
        # H_new = (num_old / num_total) * H_old + (2 / num_total) * X_new^T X_new
        beta = self.num_samples / (self.num_samples + num_new_samples)
        alpha = 2.0 / (self.num_samples + num_new_samples)
        self.H.addmm_(inp.T, inp, beta=beta, alpha=alpha)

        self.num_samples += num_new_samples

    @torch.no_grad()
    def reset(self) -> None:
        """
        Reset Hessian and calibration counters. Keeps bound layer metadata.
        """
        self.H = None
        self.num_samples = 0
        self.issue_zero_samples = False
        self.issue_nan_hessian = False
        self.issue_non_invertible = False

        if self.layer is not None:
            self._init_from_layer(self.layer)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Hessian prep + weight prep (MoE-Quant style, simplified)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def quantization_pre_step(self) -> None:
        """
        Hessian regularization and weight reshaping before GPTQ.

        - Ensures H is valid SPD (handles zero-sample / NaN cases).
        - Builds a working weight in transposed (C, R) layout for the solver,
          using a compute-friendly dtype (never fp8).
        """
        if self.layer is None:
            raise RuntimeError("GPTQ.quantization_pre_step() requires a bound layer.")

        if self.d_col is None or self.d_row is None:
            self._init_from_layer(self.layer)

        # 1) Hessian preparation
        if self.H is None:
            # No calibration data: fall back to identity Hessian
            self.H = torch.eye(self.d_col, device=self.W_device, dtype=torch.float32)
            self.issue_zero_samples = True

        # Replace H by identity if NaNs are present
        if torch.isnan(self.H).any().item():
            self.H = torch.eye(self.d_col, device=self.W_device, dtype=torch.float32)
            self.issue_nan_hessian = True

        # Pruned channels: diag == 0
        diag = torch.diag(self.H)
        pruned_ids = (diag == 0)
        if pruned_ids.any():
            self.H[pruned_ids, pruned_ids] = 1.0

        # 2) Weight preparation
        # Decide compute dtype: match layer weight unless it's fp8, then use fp16.
        param = self.layer.weight.detach()
        param_dtype = param.dtype
        float8_e4m3 = getattr(torch, "float8_e4m3fn", None)
        float8_e5m2 = getattr(torch, "float8_e5m2", None)

        compute_dtype = param_dtype
        if param_dtype in (float8_e4m3, float8_e5m2):
            compute_dtype = torch.float16

        if compute_dtype != param_dtype:
            W_param = param.to(dtype=compute_dtype)
        else:
            W_param = param

        if isinstance(self.layer, _ConvNd):
            # Flatten kernel dims; use metadata d_row/d_col from _init_from_layer
            W_flat = W_param.reshape(self.d_row, -1)
        else:
            # Linear: (out_features, in_features)
            W_flat = W_param.reshape(self.d_row, self.d_col)

        # Zero out pruned columns in the working weight
        W_flat[:, pruned_ids] = 0.0

        # Store working weight in transposed (C, R) layout to feed solver directly
        W_t = W_flat.transpose(0, 1).contiguous()  # (C, R) = (d_col, d_row)

        # Keep only the transposed working copy to avoid extra [R, C] buffers
        self.W = W_t
        self.d_col, self.d_row = W_t.shape
        self.W_device = W_t.device
        self.W_dtype = W_t.dtype
        self.W_shape = W_t.shape

        # Drop intermediates ASAP
        del W_flat
        del W_param

    # ------------------------------------------------------------------ #
    # Hessian inverse (actually: Cholesky(H^{-1})) using low-hanging fruit
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _get_hessian_inverse_cholesky(self) -> torch.Tensor:
        if self.H is None:
            raise RuntimeError("Hessian is None; call update() or quantization_pre_step() first.")
        if self.W is None:
            raise RuntimeError("Working weight W is None; call quantization_pre_step() first.")
        if self.d_col is None:
            raise RuntimeError("d_col is None; layer metadata not initialized.")

        C = self.d_col
        H = self.H  # [C, C], fp32
        w = self.W  # [C, R]

        if H.shape != (C, C):
            raise RuntimeError(f"Hessian shape {H.shape} does not match (C,C)=({C},{C}).")

        # Dead channels handling (same as before)
        zero_cols = torch.nonzero(w.eq(0).all(dim=1), as_tuple=False).view(-1)
        if zero_cols.numel() > 0:
            H[zero_cols, :] = 0.0
            H[:, zero_cols] = 0.0
            H[zero_cols, zero_cols] = 1.0

        # Damping
        diag_H = H.diagonal()
        damp = self.rel_damp * diag_H.mean()
        diag_H.add_(damp)

        try:
            if self.algorithm == "babai":
                # A = Chol(H)^T (upper-tri), used by solver_babai.cu
                out = torch.linalg.cholesky(H, upper=False).T
            else:
                # out = Chol(H^{-1}) (upper-tri) used by GPTQ solver
                L = torch.linalg.cholesky(H, upper=False)
                H_inv = torch.cholesky_inverse(L, upper=False)
                out = torch.linalg.cholesky(H_inv, upper=True)
                del L, H_inv

                # Keep your existing MoE-Quant style row-normalization ONLY for GPTQ path
                

        except Exception:
            self.issue_non_invertible = True
            out = torch.eye(C, device=H.device, dtype=torch.float32)

        #diag_u = out.diagonal()
        #diag_u = torch.where(diag_u == 0, torch.ones_like(diag_u), diag_u)
        #out = out / diag_u.unsqueeze(-1)

        # Match dtype to working weight (keeps memory sane)
        w_dtype = self.W.dtype if self.W is not None else out.dtype
        if out.dtype != w_dtype:
            out = out.to(dtype=w_dtype)

        # free Hessian
        self.H = None
        return out



    # ------------------------------------------------------------------ #
    # High-level GPTQ layer quantization using 4Bit-Forge solver
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _quantize_layer(self, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Internal: Quantize the bound layer's weight using GPTQ.

        Returns
        -------
        qweight_t: (d_col, d_row) uint8, transposed quantized weights
        qmeta:     (d_col, num_groups, 4) uint8
        maxq:      scalar tensor
        """
        if self.layer is None:
            raise RuntimeError("GPTQ._quantize_layer() requires a bound layer.")
        if self.W is None:
            raise RuntimeError("Working weight W is None; call quantization_pre_step() first.")
        if self.d_row is None or self.d_col is None:
            self._init_from_layer(self.layer)

        # Working weight is already in (C, R) layout from quantization_pre_step
        W_t = self.W  # (C, R)
        C, R = W_t.shape

        # Determine group size along R dimension
        group_size = self.group_size or R

        # Build quantization grid -> qmeta4
        device = W_t.device
        impl = "cuda" if device.type == "cuda" else "cuda"  # CUDA kernels by default

        qmeta, maxq, pad = self.build_quant_grid(
            weight=W_t,
            group_size=group_size,
            bits=bits,
            symmetric=self.sym,
            mode=self.quantization_scale,
            impl=impl,
        )

        # Compute H^{-1} Cholesky factor (dtype-matched to working weight / compute dtype)
        H_inv_cho = self._get_hessian_inverse_cholesky()

        # Solve GPTQ using 4Bit-Forge solver
        qweight_t = self.solver(
            weight=W_t,
            hessian_inv=H_inv_cho,
            qmeta=qmeta,
            maxq=maxq,
            group_size=group_size,
            bits=bits,
        )

        return qweight_t, qmeta, maxq

    @torch.no_grad()
    def quantize(self, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full GPTQ pipeline for a bound layer:

        1) `quantization_pre_step()` to prep H and working weight.
        2) `_get_hessian_inverse_cholesky()` to get chol(H^{-1}).
        3) `build_quant_grid()` to get qmeta4.
        4) `solver()` to run GPTQ and return quantized weights.

        Returns
        -------
        qweight: (out_features, in_features) uint8
        qmeta:   (in_features, num_groups, 4) uint8
        maxq:    scalar tensor (2**bits - 1)
        """
        self.quantization_pre_step()
        qweight_t, qmeta, maxq = self._quantize_layer(bits)

        # We no longer need the working weight after quantization completes
        self.W = None

        # Convert back to (out_features, in_features)
        qweight = qweight_t.transpose(0, 1).contiguous()
        return qweight, qmeta, maxq

    # ================================================================== #
    #                       Existing 4Bit-Forge API                      #
    #          (build_quant_grid + solver + qmeta helpers)               #
    # ================================================================== #

    @torch.no_grad()
    def build_quant_grid(
        self,
        weight: torch.Tensor,      # (C, R) transposed weight matrix
        group_size: int,
        bits: int,
        symmetric: bool = False,
        mode: str = "absmax",      # "absmax" or "mse"
        quant_max_shrink: float = 0.2,
        quant_n_grid: int = 100,
        quant_norm: float = 2.4,
        impl: str = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Build *groupwise* quantization metadata in packed qmeta4 format.

        Args
        ----
        weight: (C, R) transposed weight matrix.
        group_size: number of columns per quant group along dim=1 (typically 128).
                    Must be a multiple of 32.
        bits: bit-width (e.g. 4, 8).
        symmetric: if True, symmetric quant with fixed mid qzero.
        mode: "absmax" (range-based only) or "mse" (range + MSE shrink search).
        quant_max_shrink: min shrink factor in MSE grid.
        quant_n_grid: number of shrink candidates.
        quant_norm: L^p norm for the MSE loss (e.g. 2.4).

        Returns
        -------
        qmeta: (C, G, 4) uint8
               G = number of groups along R = ceil(R / group_size)
               Layout per group (4 bytes):
                 [0..1] : int16 log2(scale) in Q8.8 (little-endian)
                 [2]    : uint8 qzero
                 [3]    : uint8 flags (currently unused)
        maxq:  scalar tensor (fp32, same device as weight) with value 2**bits - 1
        pad:   int, number of padded columns added at the end of dim=1 internally
               (you pass original weight (C, R) to solver; it will ignore pad)
        """
        assert weight.ndim == 2
        C, R = weight.shape
        device = weight.device

        if group_size is None or group_size <= 0 or group_size > R:
            group_size = R
        if group_size % 32 != 0:
            raise ValueError(f"group_size must be a multiple of 32, got {group_size}")

        # Keep original dtype on CUDA (fp16/bf16/fp32/fp8), fp32 reference on CPU.
        if device.type == "cuda":
            W = weight.contiguous()
        else:
            W = weight.to(torch.float32).contiguous()

        # Compute number of groups along R and pad last group if needed
        num_groups = (R + group_size - 1) // group_size
        padded_R = num_groups * group_size
        pad = padded_R - R

        if pad > 0:
            W_pad = F.pad(W, (0, pad))
        else:
            W_pad = W

        # Reshape into groups: (C, num_groups, group_size) -> (C*num_groups, group_size)
        W_groups = W_pad.view(C, num_groups, group_size)
        x_groups = W_groups.reshape(-1, group_size)  # [G_total, group_size]

        # Unified device-dispatch for CPU / CUDA
        qmeta_flat, maxq = self._build_quant_grid_groups(
            x_groups=x_groups,
            bits=bits,
            symmetric=symmetric,
            mode=mode,
            quant_max_shrink=quant_max_shrink,
            quant_n_grid=quant_n_grid,
            quant_norm=quant_norm,
            impl=impl,
        )

        # Reshape qmeta back to (C, num_groups, 4)
        qmeta = qmeta_flat.view(C, num_groups, 4)

        return qmeta, maxq, pad

    @torch.no_grad()
    def solver(
        self,
        weight: torch.Tensor,        # (C, R), transposed weight, modified in-place
        hessian_inv: torch.Tensor,   # (C, C), GPTQ: Chol(H^{-1}); Babai: A = Chol(H)^T
        qmeta: torch.Tensor,         # (C, G, 4) uint8, packed groupwise meta
        maxq: torch.Tensor | None,   # scalar tensor (kept for API; we infer from bits)
        group_size: int,
        bits: int,
    ) -> torch.Tensor:
        """
        GPTQ solver (groupwise, qmeta4-based).

        CUDA fast-path (gptq_solver) + Python reference CPU fallback.
        """
        # CUDA only for Babai
        if weight.device.type == "cuda":
            if self.algorithm == "babai":
                return cuda_kernels.babai_solver(
                    weight,
                    hessian_inv,  # A = Chol(H)^T
                    qmeta,
                    group_size,
                    bits,
                    32,  # block_size (<=32)
                )
            else:
                return cuda_kernels.gptq_solver(
                    weight,
                    hessian_inv,  # Chol(H^{-1})
                    qmeta,
                    group_size,
                    bits,
                    32,
                )

        if self.algorithm == "babai":
            raise RuntimeError("Babai solver is CUDA-only (no CPU reference path).")

        # ------------------------------------------------------------------
        # 2. CPU Reference Implementation
        # ------------------------------------------------------------------
        assert weight.ndim == 2
        C, R = weight.shape
        assert hessian_inv.shape == (C, C)
        assert qmeta.ndim == 3 and qmeta.size(0) == C and qmeta.size(2) == 4
        assert group_size > 0

        device = weight.device
        w_dtype = weight.dtype

        num_groups = qmeta.size(1)

        # Derive maxq from bits to stay in sync with CUDA
        maxq_bits = (1 << bits) - 1
        maxq_val = float(maxq_bits)

        # Optional sanity check against provided maxq tensor
        try:
            if maxq is not None:
                diff = abs(float(maxq.item()) - maxq_val)
                if diff > 1e-3:
                    pass
        except Exception:
            pass

        # Work in float32 internally; keep original dtype for final weight update
        W = weight.to(torch.float32).contiguous()
        Hcho = hessian_inv.to(torch.float32).contiguous()

        qweight = torch.empty_like(weight, dtype=torch.uint8, device=device)

        INV256 = 1.0 / 256.0
        block_size = 32

        for block_start in range(0, C, block_size):
            block_end = min(block_start + block_size, C)
            B = block_end - block_start

            # 1) Quantize this block and accumulate delta = W_old - W_quant
            delta_block = torch.zeros(B, R, dtype=torch.float32, device=device)

            for row_offset, j in enumerate(range(block_start, block_end)):
                # Decode qmeta row j: (G, 4) -> per-group scales/inv_scales/qzeros
                row_meta = qmeta[j].to(torch.int32)  # (G, 4)

                lo = row_meta[:, 0]
                hi = row_meta[:, 1]
                log2_q88 = lo | (hi << 8)

                # sign-extend from int16
                log2_q88 = torch.where(log2_q88 >= 32768, log2_q88 - 65536, log2_q88)

                log2_scale = log2_q88.float() * INV256
                scales = torch.exp2(log2_scale)        # (G,)
                inv_scales = torch.exp2(-log2_scale)   # (G,)

                qzeros_u8 = row_meta[:, 2]
                flags = row_meta[:, 3]

                is_sym = (flags & 1) != 0
                sym_q0 = (maxq_val + 1.0) * 0.5
                qzeros = torch.where(
                    is_sym,
                    sym_q0,
                    qzeros_u8.float(),
                )  # (G,)

                for g in range(num_groups):
                    start = g * group_size
                    if start >= R:
                        break
                    end = min(start + group_size, R)
                    if start >= end:
                        continue

                    s = scales[g]
                    inv_s = inv_scales[g]
                    q0 = qzeros[g]

                    # x_old
                    x = W[j, start:end].clone()

                    # q = clamp(round(x * inv_s + q0))
                    biased = x * inv_s + q0
                    q = torch.round(biased).clamp_(0.0, maxq_val)

                    # y = (q - q0) * s
                    y = (q - q0) * s

                    # CUDA delta: x_old - y
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
            A_lower = H_block.t().contiguous()                            # [B, B], lower-tri

            Delta_J = delta_block                    # [B, R]
            Delta_T = Delta_J.t().contiguous()       # [R, B]
            E_T = torch.empty_like(Delta_T)          # [R, B]

            # Forward substitution: A_lower * x = b per row of Delta_T
            for r in range(R):
                b = Delta_T[r]  # (B,)
                x_vec = torch.empty(B, dtype=torch.float32, device=device)
                for i in range(B):
                    if i == 0:
                        s = 0.0
                    else:
                        s = torch.dot(A_lower[i, :i], x_vec[:i])
                    diag = A_lower[i, i]
                    x_vec[i] = (b[i] - s) / diag
                E_T[r] = x_vec

            E_J = E_T.t().contiguous()  # [B, R]

            # 3) Tail update: W_tail -= H_cross^T @ E_J
            H_cross = Hcho[block_start:block_end, block_end:C]  # [B, C_tail]
            if H_cross.numel() > 0:
                W_tail = W[block_end:C, :].to(torch.float32)    # [C_tail, R]
                # H_cross^T: [C_tail, B], E_J: [B, R]
                W_tail = W_tail - H_cross.t().mm(E_J)
                W[block_end:C, :] = W_tail

        # After all updates, copy solver buffer back to original dtype
        weight.copy_(W.to(w_dtype))

        return qweight

    # ------------------------------------------------------------------ #
    # Unified CPU / GPU quant-grid builder for grouped scales/qzeros
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _build_quant_grid_groups(
        self,
        x_groups: torch.Tensor,    # (G_total, group_size)
        bits: int,
        symmetric: bool,
        mode: str,
        quant_max_shrink: float,
        quant_n_grid: int,
        quant_norm: float,
        impl: str = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Unified quant-grid builder.

        - On CUDA: uses fused kernels (build_group_meta_packed + optional MSE refine).
        - On CPU: uses a slow Torch reference path + qmeta4 packing.
        """
        assert x_groups.ndim == 2
        device = x_groups.device
        mode_l = mode.lower()

        # ---------------------- CUDA path ----------------------
        if x_groups.device.type == "cuda":
            backend = cuda_kernels if impl == "cuda" else triton_kernels

            build_group_meta_packed_fn = backend.build_group_meta_packed
            qmeta_bytes, maxq = build_group_meta_packed_fn(
                x_groups,
                bits,
                symmetric,
            )

            if mode_l == "mse":
                # p must be float32 for cudaMemcpyToSymbol(c_p) inside kernels.
                p = torch.linspace(
                    1.0,
                    quant_max_shrink,
                    quant_n_grid,
                    dtype=torch.float32,
                    device=device,
                )
                mse_scale_groups_packed_fn = backend.mse_scale_groups_packed
                qmeta_bytes = mse_scale_groups_packed_fn(
                    x_groups,
                    p,
                    qmeta_bytes,
                    float(maxq.item()),
                    float(quant_norm),
                )

            return qmeta_bytes, maxq

        # ---------------------- CPU reference path ----------------------
        # Expect fp32 on CPU; caller already enforces this.
        x = x_groups.to(torch.float32)
        G, group_size = x.shape

        # Range-based meta (same semantics as CUDA build_group_meta_packed)
        maxq_val = (1 << bits) - 1
        maxq = torch.tensor(maxq_val, dtype=torch.float32, device=device)

        xmin = x.min(dim=-1).values  # (G,)
        xmax = x.max(dim=-1).values  # (G,)

        eps = 1e-12
        if symmetric:
            amax = torch.maximum(xmin.abs(), xmax.abs())
            scale = (2.0 / maxq) * amax + eps               # (G,)
            qzero = torch.full_like(scale, ((maxq + 1.0) * 0.5).item())
        else:
            scale = (xmax - xmin) / maxq + eps              # (G,)
            qzero = torch.round(-xmin / scale).clamp_(0.0, float(maxq_val))

        # Optional MSE refinement on CPU (slow, but used only as reference)
        if mode_l == "mse":
            p = torch.linspace(
                1.0,
                quant_max_shrink,
                quant_n_grid,
                dtype=x.dtype,
                device=device,
            )

            new_scale = torch.empty_like(scale)
            maxq_f = float(maxq_val)

            for g in range(G):
                xg = x[g]               # (group_size,)
                base_s = float(scale[g].item())
                q0 = float(qzero[g].item())

                best_loss = float("inf")
                best_s = base_s

                for k in range(p.numel()):
                    s = base_s * float(p[k].item())
                    if s <= 0.0:
                        continue

                    q = torch.round(xg / s + q0)
                    q.clamp_(0.0, maxq_f)
                    y = (q - q0) * s
                    diff = (y - xg).abs()

                    loss = diff.pow(quant_norm).sum().item()
                    if loss < best_loss:
                        best_loss = loss
                        best_s = s

                new_scale[g] = best_s

            scale = new_scale

        # Pack into qmeta4 bytes
        qmeta_bytes = self._encode_qmeta_groups(scale, qzero)
        return qmeta_bytes, maxq

    # ------------------------------------------------------------------ #
    # qmeta encode/decode helpers
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _encode_qmeta_groups(
        self,
        scale_g: torch.Tensor,   # (G,) float32
        qzero_g: torch.Tensor,   # (G,) float32
    ) -> torch.Tensor:
        """
        Encode per-group (scale, qzero) into packed 4-byte format:

          log2_scale_q88 = round(log2(scale) * 256)
          bytes[0] = low  8 bits
          bytes[1] = high 8 bits
          bytes[2] = qzero (uint8)
          bytes[3] = flags (currently 0)
        """
        assert scale_g.ndim == 1 and qzero_g.ndim == 1
        assert scale_g.shape == qzero_g.shape
        device = scale_g.device

        # avoid log of zero
        eps = 1e-12
        s = torch.clamp(scale_g, min=eps)
        log2_fp = torch.log2(s)
        log2_q88 = torch.round(log2_fp * 256.0).to(torch.int16)  # (G,)

        # split into low/high bytes (little-endian)
        lo = (log2_q88 & 0xFF).to(torch.uint8)          # (G,)
        hi = ((log2_q88 >> 8) & 0xFF).to(torch.uint8)   # (G,)

        qzero_u8 = qzero_g.round().clamp(0, 255).to(torch.uint8)

        qmeta = torch.empty(scale_g.shape[0], 4, dtype=torch.uint8, device=device)
        qmeta[:, 0] = lo
        qmeta[:, 1] = hi
        qmeta[:, 2] = qzero_u8
        qmeta[:, 3] = 0  # flags reserved

        return qmeta

    @torch.no_grad()
    def _decode_qmeta_groups(
        self,
        qmeta_bytes: torch.Tensor,         # (G, 4) uint8
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode packed 4-byte metadata into (scale_g, qzero_g).

        Layout per group (4 bytes):
          [0..1] : int16 log2(scale) in Q8.8 (little-endian)
          [2]    : uint8 qzero
          [3]    : uint8 flags (unused)
        """
        assert qmeta_bytes.ndim == 2 and qmeta_bytes.size(1) == 4
        assert qmeta_bytes.dtype == torch.uint8

        device = qmeta_bytes.device

        lo = qmeta_bytes[:, 0].to(torch.int16)
        hi = qmeta_bytes[:, 1].to(torch.int16)
        log2_q88 = lo | (hi << 8)                              # (G,)
        log2_fp = log2_q88.to(torch.float32) / 256.0           # (G,)
        scale = torch.exp2(log2_fp)                            # (G,)

        qzero = qmeta_bytes[:, 2].to(torch.float32)            # (G,)

        if dtype is not None:
            scale = scale.to(dtype=dtype).to(torch.float32)

        return scale.to(device=device), qzero.to(device=device)