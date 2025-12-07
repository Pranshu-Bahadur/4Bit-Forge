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
        impl: str = 'cuda'
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

        # Branch early: GPU vs CPU so we don't accidentally upcast on CUDA.
        if device.type == "cuda":
            # Keep original dtype (fp16/bf16/fp32/fp8) to avoid extra 64–256 MB copies.
            W = weight.contiguous()
        else:
            # CPU reference runs in fp32.
            W = weight.to(torch.float32).contiguous()

        # Compute number of groups along R and pad last group if needed
        num_groups = (R + group_size - 1) // group_size
        padded_R = num_groups * group_size
        pad = padded_R - R

        if pad > 0:
            # pad at the end of last dim
            W_pad = F.pad(W, (0, pad))
        else:
            W_pad = W

        # Reshape into groups: (C, num_groups, group_size) -> (C*num_groups, group_size)
        W_groups = W_pad.view(C, num_groups, group_size)
        x_groups = W_groups.reshape(-1, group_size)  # [G_total, group_size], G_total = C * num_groups

        if device.type == "cuda":
            qmeta_flat, maxq = self._build_quant_grid_gpu(
                x_groups=x_groups,
                bits=bits,
                symmetric=symmetric,
                mode=mode,
                quant_max_shrink=quant_max_shrink,
                quant_n_grid=quant_n_grid,
                quant_norm=quant_norm,
                impl=impl
            )
        else:
            qmeta_flat, maxq = self._build_quant_grid_cpu(
                x_groups=x_groups,
                bits=bits,
                symmetric=symmetric,
                mode=mode,
                quant_max_shrink=quant_max_shrink,
                quant_n_grid=quant_n_grid,
                quant_norm=quant_norm,
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
        if weight.is_cuda:
            return cuda_kernels.gptq_solver(
                weight,
                hessian_inv,
                qmeta,
                group_size,
                bits,
                0  # block_size=0 -> let kernel infer from SMEM limits
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
                    # You can log or warn here if you want
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
            # qmeta[j]: (G, 4) uint8 -> int32 for bit ops
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
            qzeros    = torch.where(is_sym,
                                    sym_qzero,
                                    qzeros_u8.float())  # (G,)

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
                    # h_tail: (C - j - 1,)
                    # e    : (group_len,)
                    # broadcast to (C - j - 1, group_len)
                    W[j + 1 :, start:end] += h_tail.unsqueeze(1) * e.unsqueeze(0)

        # After all updates, copy solver buffer back to original dtype
        weight.copy_(W.to(w_dtype))

        return qweight


    # ---------- GPU helpers: qmeta4 path ----------

    @torch.no_grad()
    def _build_quant_grid_gpu(
        self,
        x_groups: torch.Tensor,    # (G_total, group_size), same dtype as weight
        bits: int,
        symmetric: bool,
        mode: str,
        quant_max_shrink: float,
        quant_n_grid: int,
        quant_norm: float,
        impl: str = 'cuda'
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        GPU path: use qmeta4 CUDA kernels, return packed metadata.
        """
        assert x_groups.is_cuda
        assert x_groups.ndim == 2
        G, group_size = x_groups.shape
        device = x_groups.device

        # 1) Initial range-based meta
        build_group_meta_packed_fn = (cuda_kernels if impl == 'cuda' else triton_kernels).build_group_meta_packed
        qmeta_bytes, maxq = build_group_meta_packed_fn(
            x_groups,
            bits,
            symmetric,
        )

        # 2) Optional MSE refinement in-place on qmeta
        if mode.lower() == "mse":
            p = torch.linspace(
                1.0,
                quant_max_shrink,
                quant_n_grid,
                dtype=torch.float32,  # must be float32 for cudaMemcpyToSymbol(c_p)
                device=device,
            )

            mse_scale_groups_packed_fn = (cuda_kernels if impl == 'cuda' else triton_kernels).mse_scale_groups_packed
            qmeta_bytes = mse_scale_groups_packed_fn(
                x_groups,
                p,
                qmeta_bytes,
                float(maxq.item()),
                float(quant_norm),
            )

        return qmeta_bytes, maxq

    # ---------- CPU helpers: emulate qmeta4 ----------

    @torch.no_grad()
    def _build_quant_grid_cpu(
        self,
        x_groups: torch.Tensor,    # (G_total, group_size) fp32
        bits: int,
        symmetric: bool,
        mode: str,
        quant_max_shrink: float,
        quant_n_grid: int,
        quant_norm: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        CPU / Torch path: emulate qmeta4 using the same semantics.
        """
        assert not x_groups.is_cuda
        assert x_groups.ndim == 2
        device = x_groups.device
        G, group_size = x_groups.shape

        # 1) Initial range-based meta per group
        scale_g, qzero_g, maxq = self._find_quantization_meta_groups(
            x_groups,
            bit_width=bits,
            symmetric=symmetric,
        )  # (G,), (G,), scalar

        # 2) Optional MSE refinement (shrinking scale)
        if mode.lower() == "mse":
            p = torch.linspace(
                1.0,
                quant_max_shrink,
                quant_n_grid,
                dtype=x_groups.dtype,
                device=device,
            )
            scale_g = self._mse_scale_groups(
                x_groups,
                p=p,
                scale=scale_g,
                qzero=qzero_g,
                maxq=maxq,
                norm=quant_norm,
            )

        # 3) Pack into qmeta4: [0..1] log2(scale) in Q8.8, [2] qzero, [3] flags=0
        qmeta_bytes = self._encode_qmeta_groups(scale_g, qzero_g)

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
        G = scale_g.shape[0]

        # avoid log of zero
        eps = 1e-12
        s = torch.clamp(scale_g, min=eps)
        log2_fp = torch.log2(s)
        log2_q88 = torch.round(log2_fp * 256.0).to(torch.int16)  # (G,)

        # split into low/high bytes (little-endian)
        lo = (log2_q88 & 0xFF).to(torch.uint8)          # (G,)
        hi = ((log2_q88 >> 8) & 0xFF).to(torch.uint8)   # (G,)

        qzero_u8 = qzero_g.round().clamp(0, 255).to(torch.uint8)

        qmeta = torch.empty(G, 4, dtype=torch.uint8, device=device)
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

    # ---------- CPU reference quant grid math (unpacked) ----------

    @torch.no_grad()
    def _find_quantization_meta_groups(
        self,
        x_groups: torch.Tensor,    # (G_total, group_size)
        bit_width: int,
        symmetric: bool = False,
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Range-based quantization meta per group over dim=-1.

        x_groups: (G, group_size)
        Returns:
          scale_g: (G,)
          qzero_g: (G,)
          maxq:    scalar tensor
        """
        assert x_groups.ndim == 2
        device = x_groups.device

        maxq_val = (1 << bit_width) - 1
        maxq = torch.tensor(maxq_val, dtype=torch.float32, device=device)

        xmin = x_groups.min(dim=-1).values  # (G,)
        xmax = x_groups.max(dim=-1).values  # (G,)

        if symmetric:
            amax = torch.maximum(xmin.abs(), xmax.abs())
            scale = (2.0 / maxq) * amax + eps
            qzero = torch.full_like(scale, ((maxq + 1.0) * 0.5).item())  # middle code
        else:
            scale = (xmax - xmin) / maxq + eps
            qzero = torch.round(-xmin / scale).clamp_(0.0, float(maxq_val))

        return scale, qzero, maxq

    @torch.no_grad()
    def _mse_scale_groups(
        self,
        x_groups: torch.Tensor,    # (G_total, group_size)
        p: torch.Tensor,           # (P,) shrink factors
        scale: torch.Tensor,       # (G_total,)
        qzero: torch.Tensor,       # (G_total,)
        maxq: torch.Tensor,        # scalar
        norm: float = 2.4,
    ) -> torch.Tensor:
        """
        Naive CPU reference for MSE-based scale refinement.

        For each group g:
          - try scale_g * p[k] for all k
          - pick the one minimizing sum(|q(x; s_k) - x|^norm)

        Returns updated `scale` (same shape).
        """
        assert x_groups.ndim == 2
        G, group_size = x_groups.shape
        assert scale.shape == (G,)
        assert qzero.shape == (G,)
        assert p.ndim == 1

        device = x_groups.device
        maxq_val = float(maxq.to(device).item())

        x = x_groups.to(torch.float32)
        scale = scale.to(torch.float32)
        qzero = qzero.to(torch.float32)
        p = p.to(torch.float32)

        new_scale = torch.empty_like(scale)

        for g in range(G):
            xg = x[g]               # (group_size,)
            base_s = scale[g].item()
            q0 = qzero[g].item()

            best_loss = float("inf")
            best_s = base_s

            for k in range(p.numel()):
                s = base_s * float(p[k].item())
                if s <= 0:
                    continue

                q = torch.round(xg / s + q0)
                q.clamp_(0.0, maxq_val)
                y = (q - q0) * s
                diff = (y - xg).abs()

                loss = diff.pow(norm).sum().item()
                if loss < best_loss:
                    best_loss = loss
                    best_s = s

            new_scale[g] = best_s

        return new_scale
