import torch
import torch.nn.functional as F
from forge.cuda import kernels as cuda_kernels
from forge.trt import kernels as triton_kernels


class GPTQ:
    """
    GPTQ grid builder + solver (qmeta4-based).

    Design:
      - Grid builder works on groups of columns (group_size, typically 128).
      - On CUDA: uses packed 4-byte metadata per group (log2(scale) in Q8.8 + qzero + flags).
      - On CPU: emulates the same qmeta4 format via pure Torch ops.
      - Solver consumes qmeta4 directly (groupwise), no full (C, R) scale/qzero tensors.
    """

    def __init__(self, device: str = ""):
        self.device = device

    # ---------- public-ish API ----------

    @torch.no_grad()
    def build_quant_grid(
        self,
        weight: torch.Tensor,      # (C, R)
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
        quant_max_shrink: min shrink factor in MSE grid (same semantics as MoE-Quant).
                          Search is over p in [1.0, quant_max_shrink].
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
        x_groups = W_groups.reshape(-1, group_size)  # [G_total, group_size], G_total = C * num_groups

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
        weight: torch.Tensor,       # (C, R), transposed weight, modified in-place
        hessian_inv: torch.Tensor,  # (C, C), inverse Hessian
        qmeta: torch.Tensor,        # (C, G, 4) uint8, packed groupwise meta
        maxq: torch.Tensor,         # scalar tensor (kept for API; we infer from bits)
        group_size: int,
        bits: int,
    ) -> torch.Tensor:
        """
        GPTQ solver (groupwise, qmeta4-based).

        CUDA fast-path (gptq_solver) + Python reference CPU fallback.

        Returns
        -------
        qweight: (C, R) uint8 quantized codes.
        """
        # ----------------------------------------------------------------------
        # 1. CUDA Fast Path
        # ----------------------------------------------------------------------
        if weight.device.type == "cuda":
            return cuda_kernels.gptq_solver(
                weight,
                hessian_inv,
                qmeta,
                group_size,
                bits,
                32,  # block_size=0 -> let kernel infer from SMEM limits
            )

        # ----------------------------------------------------------------------
        # 2. CPU Reference Implementation (Matches Kernel Math)
        # ----------------------------------------------------------------------
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
                    # could warn/log here if desired
                    pass
        except Exception:
            # If maxq isn't a proper scalar tensor, just ignore it
            pass

        # Work in float32 for solver; keep original dtype for final copy-out
        W = weight.to(torch.float32).contiguous()
        Hinv = hessian_inv.to(torch.float32).contiguous()

        # Output codes
        qweight = torch.empty_like(weight, dtype=torch.uint8, device=device)

        INV256 = 1.0 / 256.0

        for j in range(C):
            # --------------------------------------------------------------
            # A. Decode qmeta for row j  -> scales, inv_scales, qzeros
            # --------------------------------------------------------------
            row_meta = qmeta[j].to(torch.int32)  # (G, 4)

            # bytes 0,1 = int16 log2_q88 (little endian)
            lo = row_meta[:, 0]
            hi = row_meta[:, 1]
            log2_q88 = lo | (hi << 8)

            # sign-extend 16-bit -> 32-bit
            log2_q88 = torch.where(log2_q88 >= 32768, log2_q88 - 65536, log2_q88)

            log2_scale = log2_q88.float() * INV256
            scales     = torch.exp2(log2_scale)        # (G,)
            inv_scales = torch.exp2(-log2_scale)       # (G,)

            qzeros_u8 = row_meta[:, 2]
            flags     = row_meta[:, 3]

            # symmetric override: if flags & 1: qzero = (maxq + 1)/2
            is_sym   = (flags & 1) != 0
            sym_qzero = (maxq_val + 1.0) * 0.5
            qzeros    = torch.where(
                is_sym,
                sym_qzero,
                qzeros_u8.float(),
            )  # (G,)

            # --------------------------------------------------------------
            # B. Quantize row j, accumulate error, propagate via Hinv
            # --------------------------------------------------------------
            if j + 1 < C:
                h_tail = Hinv[j, j + 1 :]   # (C - j - 1,)
            else:
                h_tail = None

            for g in range(num_groups):
                start = g * group_size
                if start >= R:
                    break
                end = min(start + group_size, R)

                s      = scales[g]
                inv_s  = inv_scales[g]
                q0     = qzeros[g]

                x = W[j, start:end]        # (group_len,)

                # 1. Quantize: q = clamp(round(x * inv_s + q0))
                biased = x * inv_s + q0
                q = torch.round(biased)
                q.clamp_(0.0, maxq_val)

                # 2. Dequantize: y = (q - q0) * s
                y = (q - q0) * s

                # 3. Error: e = y - x
                e = y - x

                # Write into working buffer + codes
                W[j, start:end]          = y
                qweight[j, start:end]    = q.to(torch.uint8)

                # 4. Propagate error to future rows: W[k] += Hinv[j, k] * e
                if h_tail is not None:
                    W[j + 1 :, start:end] += h_tail.unsqueeze(1) * e.unsqueeze(0)

        # After all updates, copy solver buffer back to original dtype
        weight.copy_(W.to(w_dtype))

        return qweight

    # ---------- unified CPU / GPU quant-grid builder ----------

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


    # ---------- qmeta encode/decode helpers ----------

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
