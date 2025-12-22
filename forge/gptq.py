# forge/gptq.py
from __future__ import annotations

from typing import Optional, Tuple, Literal, Dict, Any
from enum import Enum

import os
import torch
import torch.nn as nn

from torch.nn.modules.conv import _ConvNd

from forge.cuda import kernels as cuda_kernels
from forge.trt import kernels as triton_kernels

try:
    import torch.distributed as dist
except Exception:
    dist = None


class QuantizationOrder(Enum):
    DEFAULT = "default"
    ACTIVATION = "activation"

# Dist (DP & TP Support)
def _dist_available_and_initialized() -> bool:
    return dist is not None and dist.is_available() and dist.is_initialized()


def _is_main_process(is_distributed: bool) -> bool:
    if not is_distributed:
        return True
    if not _dist_available_and_initialized():
        return True
    return dist.get_rank() == 0


def _all_reduce_avg_(x: torch.Tensor, group=None) -> None:
    if not _dist_available_and_initialized():
        return x
    world = dist.get_world_size(group=group)
    if world <= 1:
        return x
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
    x.div_(float(world))

def _all_reduce_sum_(x: torch.Tensor, group=None) -> torch.Tensor: # Fix type hint
    if not _dist_available_and_initialized():
        return x  # <--- MUST RETURN x
    world = dist.get_world_size(group=group)
    if world <= 1:
        return x  # <--- MUST RETURN x
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
    return x


def _is_float8(dtype: torch.dtype) -> bool:
    # robust across torch builds
    return str(dtype).startswith("torch.float8")


def _maybe_set_dynamo_suppress_errors(enable: bool) -> None:
    if not enable:
        return
    try:
        import torch._dynamo  # type: ignore
        torch._dynamo.config.suppress_errors = True
    except Exception:
        pass


def _maybe_compile(
    fn,
    enabled: bool,
    mode: str,
    fullgraph: bool,
    dynamic: bool,
):
    if not enabled:
        return fn
    if not hasattr(torch, "compile"):
        return fn
    try:
        sig = torch.compile  # type: ignore[attr-defined]
        # torch.compile signature varies a bit across versions; pass what we can.
        kwargs = {}
        try:
            import inspect
            ps = inspect.signature(sig).parameters
            if "mode" in ps:
                kwargs["mode"] = mode
            if "fullgraph" in ps:
                kwargs["fullgraph"] = fullgraph
            if "dynamic" in ps:
                kwargs["dynamic"] = dynamic
        except Exception:
            pass
        return torch.compile(fn, **kwargs)  # type: ignore[misc]
    except Exception:
        return fn


class GPTQ:
    """
    4Bit-Forge GPTQ grid builder + solver (qmeta4-based) + MoE-Quant-style
    Hessian accumulation & layer-bound quantization wrapper.

    Conventions (4BF)
    -----------------
    - Working weight is stored as transposed (C, R):
        C = input dim (Hessian dim), R = output dim
      i.e. original nn.Linear.weight is (R, C), we work with W_t = (C, R).

    - qmeta is packed qmeta4:
        qmeta: (C, num_groups_along_R, 4) uint8

    - Solver expects:
        algorithm == "gptq": hessian_inv is U where H^{-1} = U^T U (upper-tri)
        algorithm == "babai": hessian_inv is A = chol(H) (upper-tri) (== L^T)

    Notes
    -----
    - This class does NOT register or rely on PyTorch hooks.
      Call `update(input_activations)` explicitly (the caller can intercept inputs however it wants).
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        layer: nn.Module | None = None,
        group_size: int | None = 128,
        sym: bool = False,
        rel_damp: float = 1e-2,
        block_size: int = 32,
        quantization_order: str = "default",
        quantization_scale: str = "absmax",
        is_distributed: bool = False,
        tied_gptq_handle: Optional["GPTQ"] = None,
        algorithm: str = "babai",
        # Optional internal torch.compile (OFF by default; caller can enable)
        torch_compile: bool = False,
        torch_compile_mode: str | None = None,
        torch_compile_fullgraph: bool = False,
        torch_compile_dynamic: bool = False,
        torch_compile_suppress_errors: bool = False,
    ):
        # Optional device hint (stateless usage)
        if device is None or device == "":
            self.device = None
        else:
            self.device = torch.device(device)

        self.algorithm = str(algorithm).lower()
        if self.algorithm not in ("gptq", "babai"):
            raise ValueError(f"Unknown algorithm={algorithm}. Expected 'gptq' or 'babai'.")

        self.layer: nn.Module | None = layer
        self.group_size = group_size
        self.sym = bool(sym)
        self.rel_damp = float(rel_damp)

        self.block_size = int(block_size) if block_size is not None else 32
        if self.block_size <= 0 or self.block_size > 32:
            raise ValueError(f"block_size must be in [1, 32], got {self.block_size}")

        self.quantization_order = QuantizationOrder(str(quantization_order).lower())
        self.quantization_scale = str(quantization_scale).lower()  # "absmax" or "mse"

        self.is_distributed = bool(is_distributed)

        # tied handle (MoE-Quant semantics)
        self.tied_gptq_handle: GPTQ | None = tied_gptq_handle
        self.num_tied_handles: int = 0
        self._owner_reset_pending: bool = False
        if tied_gptq_handle is not None:
            tied_gptq_handle.num_tied_handles += 1

        #TieHandle Owner Fields
        self._hessian_prepared = False
        self._h_factor_fp32 = None
        self._perm = None
        self._hfactor_cache = None

        # Hessian & calibration state
        self.H: torch.Tensor | None = None
        self.num_samples: torch.Tensor = torch.zeros((), device=self.device, dtype=torch.long)

        # Working weight (transposed (C,R)) during quant
        self.W: torch.Tensor | None = None
        self.d_row: int | None = None  # out_features (R)
        self.d_col: int | None = None  # in_features  (C)
        self.W_device: torch.device | None = None
        self.W_dtype: torch.dtype | None = None
        self.W_shape: torch.Size | None = None

        # Conv unfold cache (avoid recreating nn.Unfold every update call)
        self._unfold: nn.Unfold | None = None

        # Pruned ids cache (bool mask on input dims)
        self._pruned_ids: torch.Tensor | None = None

        # Issue flags
        self.issue_zero_samples: bool = False
        self.issue_nan_hessian: bool = False
        self.issue_non_invertible: bool = False

        # torch.compile (internal)
        _maybe_set_dynamo_suppress_errors(torch_compile_suppress_errors)
        self._tc_enabled = False #TODO remove torch.compile #bool(torch_compile) or (os.getenv("FORGE_GPTQ_COMPILE", "0") == "1")
        self._tc_mode = os.getenv("FORGE_GPTQ_COMPILE_MODE", torch_compile_mode)
        self._tc_fullgraph = bool(torch_compile_fullgraph)
        self._tc_dynamic = bool(torch_compile_dynamic)

        self._compiled_addmm_ = None  # lazily created

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
            W_flat = W.flatten(1, -1)  # (out, in*kH*kW)
            self.d_row = int(W_flat.shape[0])
            self.d_col = int(W_flat.shape[1])
        else:
            if W.ndim != 2:
                raise TypeError("Expected weight to be 2D for non-conv layers.")
            self.d_row, self.d_col = int(W.shape[0]), int(W.shape[1])

        # unfold cache for convs
        if isinstance(layer, _ConvNd):
            self._unfold = nn.Unfold(
                layer.kernel_size,
                dilation=layer.dilation,
                padding=layer.padding,
                stride=layer.stride,
            )
        else:
            self._unfold = None

    def has_hessian_issues(self) -> bool:
        return any([self.issue_zero_samples, self.issue_nan_hessian, self.issue_non_invertible])

    @property
    def tokens_collected(self) -> int:
        return int(self.num_samples.item())
    
    def _owner(self) -> "GPTQ":
        return self.tied_gptq_handle or self


    # ------------------------------------------------------------------ #
    # Internal compiled kernel (optional)
    # ------------------------------------------------------------------ #
    def _get_compiled_addmm_(self):
        if self._compiled_addmm_ is not None:
            return self._compiled_addmm_

        def _addmm_update_(H: torch.Tensor, X: torch.Tensor, beta: float, alpha: float):
            # X is (N, C). We update H in-place.
            H.addmm_(X.transpose(0, 1), X, beta=beta, alpha=alpha)
            return H

        self._compiled_addmm_ = _maybe_compile(
            _addmm_update_,
            enabled=self._tc_enabled,
            mode=self._tc_mode,
            fullgraph=self._tc_fullgraph,
            dynamic=self._tc_dynamic,
        )
        return self._compiled_addmm_

    # ------------------------------------------------------------------ #
    # MoE-Quant–style Hessian accumulation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def update(self, input: torch.Tensor) -> None:
        """
        Update the estimate of the Hessian matrix from a batch of layer inputs.

        If tied_gptq_handle is set, we delegate accumulation to the tied handle
        (saves duplicate X^T X work when users call update() on multiple handles).
        """
        if self.layer is None:
            raise RuntimeError("GPTQ.update() requires a bound layer (pass layer=... in __init__).")
        if self.d_col is None:
            self._init_from_layer(self.layer)

        if self.tied_gptq_handle is not None:
            self.tied_gptq_handle.update(input)
            # mirror pointers/counters (keeps tokens_collected sensible)
            self.H = self.tied_gptq_handle.H
            self.num_samples = self.tied_gptq_handle.num_samples
            return

        # Create H on the same device as inputs (caller controls where update() runs)
        if self.H is None:
            self.H = torch.zeros((self.d_col, self.d_col), device=input.device, dtype=torch.float32)

        if isinstance(self.layer, nn.Linear):
            # input: (..., C) -> (N, C)
            inp2d = input.reshape(-1, input.shape[-1])
        elif isinstance(self.layer, _ConvNd):
            if self._unfold is None:
                self._init_from_layer(self.layer)
            # (N, C*k*k, L) -> (N*L, C*k*k)
            u = self._unfold(input)  # (N, K, L)
            inp2d = u.transpose(1, 2).reshape(-1, u.shape[1])
        else:
            raise TypeError("GPTQ.update() supports nn.Linear and ConvNd layers only.")

        # Important: cast AFTER any caller-side token subsampling/capping to minimize conversion cost.
        if inp2d.dtype != torch.float32:
            inp2d = inp2d.float()

        n_new = int(inp2d.shape[0])
        if n_new <= 0:
            return

        # MoE-Quant style EMA update
        ns = int(self.num_samples.item())          # scalar python int
        total = float(ns + n_new)
        beta = float(ns) / total
        alpha = 2.0 / total


        
        self.H.addmm_(inp2d.transpose(0, 1), inp2d, beta=beta, alpha=alpha)

        self.num_samples += n_new

    @torch.no_grad()
    def reset(self) -> None:
        """
        MoE-Quant reset semantics with tied handles:

        - Tied handle:
            decrements owner's counter; when owner's count reaches 0, owner's H+num_samples are cleared.

        - Owner handle:
            if tied handles still exist, mark reset pending and keep H/num_samples until last tied handle resets.
            else clear immediately.
        """
        self.W = None
        self._pruned_ids = None

        if self.layer is not None:
            self._init_from_layer(self.layer)

        if self.tied_gptq_handle is not None:
            owner = self.tied_gptq_handle
            owner.num_tied_handles -= 1
            if owner.num_tied_handles <= 0:
                owner.num_tied_handles = 0
                owner.H = None
                owner.num_samples = torch.zeros((), device=owner.device, dtype=torch.long)
                owner._owner_reset_pending = False
                owner._hfactor_cache = None
            # tied handle clears its own view
            self.H = None
            self.num_samples = torch.zeros((), device=self.device, dtype=torch.long)
        else:
            if self.num_tied_handles > 0:
                self._owner_reset_pending = True
                # keep H/num_samples (shared) until last tied handle is done
            else:
                self.H = None
                self.num_samples = torch.zeros((), device=self.device, dtype=torch.long)
                self._owner_reset_pending = False
                self._hfactor_cache = None

        self.issue_zero_samples = False
        self.issue_nan_hessian = False
        self.issue_non_invertible = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Hessian prep + weight prep (MoE-Quant style, but 4BF layout)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _prepare_hessian_once(self, *, group=None):
        assert self.tied_gptq_handle is None
        if getattr(self, "_hessian_prepared", False):
            return

        # 0) zero-sample / missing Hessian -> identity
        if self.H is None or (self.num_samples is not None and int(self.num_samples.item()) == 0):
            C = self.d_col if self.d_col is not None else (self.H.shape[0] if self.H is not None else None)
            if C is None:
                raise RuntimeError("Cannot infer Hessian size (d_col unset and H is None).")
            dev = self.W_device if self.W_device is not None else self.layer.weight.device
            self.H = torch.eye(C, device=dev, dtype=torch.float32)
            self._pruned_ids = None
            self.issue_zero_samples = True
            self._h_perm = None
            self._hessian_prepared = True
            return

        # 1) distributed combine (weighted)
        if self.is_distributed and _dist_available_and_initialized():
            # ensure scalar tensor
            n = self.num_samples
            if not torch.is_tensor(n):
                raise RuntimeError("num_samples must be a tensor if distributed combine is enabled.")
            n_fp = n.to(device=self.H.device, dtype=self.H.dtype)  # fp32 scalar

            Hw = self.H * n_fp
            #dist.all_reduce(Hw, op=dist.ReduceOp.SUM, group=group)
            #dist.all_reduce(n_fp, op=dist.ReduceOp.SUM, group=group)
            Hw = _all_reduce_sum_(Hw, group)
            n_fp = _all_reduce_sum_(n_fp, group)

            n_den = n_fp.clamp_min(1.0)
            self.H = Hw / n_den
            self.num_samples = n_fp.to(dtype=n.dtype)

        # 2) NaN guard
        if torch.isnan(self.H).any().item():
            C = self.H.shape[0]
            self.H = torch.eye(C, device=self.H.device, dtype=self.H.dtype)
            self._pruned_ids = None
            self.issue_nan_hessian = True
            self._h_perm = None
            self._hessian_prepared = True
            return

        # 3) prune ids once
        diag = self.H.diagonal()
        pruned = (diag == 0)
        self._pruned_ids = pruned
        if pruned.any():
            self.H[pruned, pruned] = 1.0  # minimal SPD salvage; full row/col zeroing happens later in factor step

        # 4) optional perm cache (if you decide to)
        if self.quantization_order == QuantizationOrder.ACTIVATION:
            self._h_perm = torch.argsort(self.H.diagonal(), descending=True)
        else:
            self._h_perm = None

        self._hessian_prepared = True

    



    @torch.no_grad()
    def quantization_pre_step(self) -> None:
        """
        - Ensures H is valid SPD (handles zero-sample / NaN).
        - Optional distributed avg-reduce of Hessian.
        - Builds working transposed weight W_t in (C, R) contiguous layout.
        """
        if self.layer is None:
            raise RuntimeError("GPTQ.quantization_pre_step() requires a bound layer.")

        if self.d_col is None or self.d_row is None:
            self._init_from_layer(self.layer)
        


        # 1) Hessian preparation
        owner = self._owner()
            # tied handle: ensure owner is prepared

        owner._prepare_hessian_once() #TODO pass in group if needed later
        
        # mirror shared state/views
        self.H = owner.H
        self.num_samples = owner.num_samples
        self._pruned_ids = owner._pruned_ids

        

        if self.H is None:
            # no samples => identity fallback
            dev = self.W_device if self.W_device is not None else (self.layer.weight.device)
            self.H = torch.eye(self.d_col, device=dev, dtype=torch.float32)
            self.issue_zero_samples = True
        else:
            if self.num_samples.item() == 0:
                self.issue_zero_samples = True
            
        self._h_factor = owner._get_hessian_factor_cached(owner._h_perm, 
                                             rel_damp=self.rel_damp, 
                                             algorithm=self.algorithm,
                                             out_dtype=None) #self.layer.weight.dtype
            

        # 2) Weight preparation
        param = self.layer.weight.detach()

        compute_dtype = param.dtype

        if isinstance(self.layer, _ConvNd):
            # (out, in*k*k) view, then transpose -> (C, R)
            w2d = param.flatten(1, -1)
            w_t_view = w2d.transpose(0, 1)  # (C, R) view
        else:
            # Linear: (R, C) -> transpose -> (C, R)
            w_t_view = param.transpose(0, 1)

        if compute_dtype != w_t_view.dtype:
            W_t = w_t_view.to(dtype=compute_dtype).contiguous()
        else:
            W_t = w_t_view.contiguous()

        if self._pruned_ids is not None and self._pruned_ids.any():
            W_t[self._pruned_ids, :] = 0.0

        self.W = W_t
        self.d_col, self.d_row = int(W_t.shape[0]), int(W_t.shape[1])
        self.W_device = W_t.device
        self.W_dtype = W_t.dtype


    # ------------------------------------------------------------------ #
    # Hessian factor for solver (optimized)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _compute_hessian_factor_fp32(
            self,
            *,
            perm: Optional[torch.Tensor],
            rel_damp: float,
            algorithm: str,
        ) -> torch.Tensor:
            """
            Computes the Hessian factor in FP32 on a working copy (never mutates self.H).
            Returns:
            - algorithm=="gptq": U = chol(H^{-1}) upper
            - algorithm=="babai": A = chol(H) upper
            """
            if self.H is None:
                raise RuntimeError("Hessian is None; call update() first.")

            # Work on a detached fp32 clone so we don't trash shared H
            H_work = self.H.detach().to(dtype=torch.float32).clone()

            #https://github.com/IST-DASLab/MoE-Quant/blob/5a3b298cfb5c475a9b6584d48b43fcebc4ddfb2f/src/gptq.py#L221

            # Apply prune mask (in the ORIGINAL basis)
            zero_cols = torch.nonzero(self.W.eq(0).all(dim=0))

            H_work[zero_cols, :] = 0.0
            H_work[:, zero_cols] = 0.0
            H_work[zero_cols, zero_cols] = 1.0

            damp = float(rel_damp) * H_work.diagonal().mean()
            H_work[range(self.d_col), range(self.d_col)] += damp

            # Apply permutation (still working copy)
            if perm is not None and self.quantization_order == QuantizationOrder.ACTIVATION:
                H_work = H_work.index_select(0, perm).index_select(1, perm)

            # Factorize (destructive on H_work)
            try:
                if self.algorithm == "babai":
                    # A = chol(H) upper: H = A^T A
                    H_work = torch.linalg.cholesky(H_work, upper=True)
                else:
                    # U = chol(H^{-1}) upper: H^{-1} = U^T U
                    H_work = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H_work)) , upper=True)      # U in-place
            except Exception:
                self.issue_non_invertible = True
                H_work = torch.eye(H_work.shape[0], device=H_work.device, dtype=torch.float32)

            # Row-normalize by diagonal
            if self.algorithm == "gptq":
                d = H_work.diagonal().clone()
                d = torch.where(d == 0, torch.ones_like(d), d)
                H_work.div_(d.unsqueeze(-1))

            return H_work
    

    def _get_hessian_factor_cached(
            self,
            perm: Optional[torch.Tensor] = None,
            *,
            rel_damp: Optional[float] = None,
            algorithm: Optional[str] = None,
            out_dtype: Optional[torch.dtype] = None,
        ) -> torch.Tensor:
            """
            Owner caches fp32 factor once; tied handles reuse it.
            If out_dtype is provided, returns a casted view for solver bandwidth.
            """
            owner = self._owner()

            if owner.H is None:
                raise RuntimeError("Owner Hessian is None; call update() / quantization_pre_step() first.")

            rel_damp = owner.rel_damp if rel_damp is None else float(rel_damp)
            algorithm = owner.algorithm if algorithm is None else algorithm

            # Treat identity perm as None
            if perm is not None:
                C = owner.d_col
                if perm.numel() == C and torch.equal(perm, torch.arange(C, device=perm.device)):
                    perm = None

            # Cache container
            cache: Optional[Dict[str, Any]] = getattr(owner, "_hfactor_cache", None)

            def _perm_equal(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> bool:
                if a is None and b is None:
                    return True
                if (a is None) != (b is None):
                    return False
                return torch.equal(a, b)  # O(C), cheap vs factorization

            cache_key: Tuple[str, float] = (algorithm, rel_damp)

            if cache is not None:
                    factor_fp32 = cache["factor_fp32"]
                    if out_dtype is not None and factor_fp32.dtype != out_dtype:
                        return factor_fp32.to(dtype=out_dtype)
                    return factor_fp32

            # Compute once on owner
            factor_fp32 = owner._compute_hessian_factor_fp32(
                perm=perm, rel_damp=rel_damp, algorithm=algorithm
            )

            owner._hfactor_cache = {
                "perm": (perm.clone() if perm is not None else None),
                "factor_fp32": factor_fp32.clone(),
            }

            if out_dtype is not None and factor_fp32.dtype != out_dtype:
                return factor_fp32.to(dtype=out_dtype)
            return factor_fp32
    

    # Depreceated
    @torch.no_grad()
    def _get_hessian_factor(self) -> torch.Tensor:
        """
        Returns:
          algorithm=="gptq": U = chol(H^{-1}) upper
          algorithm=="babai": A = chol(H) upper

        Notes:
        - Runs in fp32 and then casts to W dtype to reduce solver bandwidth.
        - Never overwrites shared Hessians with cholesky factors.
        """
        if self.H is None:
            raise RuntimeError("Hessian is None; call update() or quantization_pre_step() first.")
        if self.W is None:
            raise RuntimeError("Working weight W is None; call quantization_pre_step() first.")

        C = self.d_col
        H = self.H

        # Only owner applies masking + damping (MoE-Quant behavior)
        if self._owner() is self:
            pruned = self._pruned_ids
            if pruned is not None and pruned.any():
                H[pruned, :] = 0.0
                H[:, pruned] = 0.0
                H[pruned, pruned] = 1.0

            damp = self.rel_damp * H.diagonal().mean()
            H.diagonal().add_(damp)

        # If Hessian can be shared, clone before factorization so we don't trash it.
        shared_h = (self.num_tied_handles > 0) or (self.tied_gptq_handle is not None)
        H_work = H.clone() if shared_h else H  # fp32 buffer

        try:
            if self.algorithm == "babai":
                # A = chol(H) upper: H = A^T A
                torch.linalg.cholesky(H_work, upper=True, out=H_work)
            else:
                # U = chol(H^{-1}) upper: H^{-1} = U^T U
                torch.linalg.inv(H_work, out=H_work)      # L in-place
                torch.linalg.cholesky(H_work, upper=True, out=H_work)       # U in-place
        except Exception:
            self.issue_non_invertible = True
            H_work = torch.eye(C, device=H.device, dtype=torch.float32)

        # Row-normalize by diagonal (in-place, with diag clone to avoid aliasing)
        d = H_work.diagonal().clone()
        d = torch.where(d == 0, torch.ones_like(d), d)
        H_work.div_(d.unsqueeze(-1))

        # Cast to W dtype for solver bandwidth
        if H_work.dtype != self.W.dtype:
            H_work = H_work.to(dtype=self.W.dtype)

        return H_work

    # ------------------------------------------------------------------ #
    # Quantize bound layer
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _quantize_layer(self, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.layer is None:
            raise RuntimeError("GPTQ._quantize_layer() requires a bound layer.")
        if self.W is None:
            raise RuntimeError("Working weight W is None; call quantization_pre_step() first.")

        bits_i = int(bits)
        W_t = self.W  # (C, R)
        C, R = int(W_t.shape[0]), int(W_t.shape[1])

        group_size = self.group_size or R
        if group_size <= 0 or group_size > R:
            group_size = R
        if group_size % 32 != 0:
            raise ValueError(f"group_size must be a multiple of 32, got {group_size}")

        # ------------------- Permutation (MoE-Quant) -------------------
        if self.quantization_order == QuantizationOrder.ACTIVATION:
            # H is fp32; diag is fp32; argsort cost is small relative to quant.
            perm = torch.argsort(self.H.diagonal(), descending=True)
        else:
            perm = torch.arange(C, device=W_t.device)

        is_identity_perm = torch.equal(perm, torch.arange(C, device=perm.device))
        if is_identity_perm:
            perm_inv = None
        else:
            perm_inv = torch.argsort(perm)

            # permute weight rows (input dims)
            W_t = W_t.index_select(0, perm)
            self.W = W_t

            #if self._owner() is self:
            #    if self._pruned_ids is not None:
            #        self._pruned_ids = self._pruned_ids.index_select(0, perm)

        # ------------------- Build qmeta4 grid -------------------
        qmeta, maxq, _pad = self.build_quant_grid(
            weight=W_t,
            group_size=group_size,
            bits=bits_i,
            symmetric=self.sym,
            mode=self.quantization_scale,
            impl="cuda",
        )

        h_factor = self._h_factor
        # ------------------- Solve -------------------
        qweight_t = self.solver(
            weight=W_t,
            hessian_inv=h_factor,
            qmeta=qmeta,
            maxq=maxq,
            group_size=group_size,
            bits=bits_i,
        )

        # ------------------- Unpermute outputs -------------------
        if not is_identity_perm:
            qweight_t = qweight_t.index_select(0, perm_inv)
            qmeta = qmeta.index_select(0, perm_inv)

        return qweight_t, qmeta, maxq

    @torch.no_grad()
    def quantize(self, bits: int | float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full pipeline for a bound layer.

        Distributed behavior (MoE-Quant-style):
        - rank0 runs quantization
        - other ranks allocate outputs and receive via broadcast
        """
        bits_i = int(bits)
        self.quantization_pre_step()

        if self.layer is None:
            raise RuntimeError("GPTQ.quantize() requires a bound layer.")

        if self.W is None:
            raise RuntimeError("quantization_pre_step() failed to set W.")

        C, R = int(self.W.shape[0]), int(self.W.shape[1])
        group_size = self.group_size or R
        if group_size <= 0 or group_size > R:
            group_size = R
        if group_size % 32 != 0:
            raise ValueError(f"group_size must be a multiple of 32, got {group_size}")
        num_groups = (R + group_size - 1) // group_size

        main = _is_main_process(self.is_distributed) or not self.is_distributed

        if main:
            qweight_t, qmeta, maxq = self._quantize_layer(bits_i)
        else:
            # allocate placeholders for broadcast
            qweight_t = torch.empty((C, R), device=self.W_device, dtype=torch.uint8)
            qmeta = torch.empty((C, num_groups, 4), device=self.W_device, dtype=torch.uint8)
            maxq = torch.empty((), device=self.W_device, dtype=torch.float32)

        if self.is_distributed and _dist_available_and_initialized():
            dist.barrier()
            dist.broadcast(qweight_t, src=0)
            dist.broadcast(qmeta, src=0)
            dist.broadcast(maxq, src=0)

        # free working weight
        self.W = None
        self._pruned_ids = None

        # return qweight in (R, C) to match original nn.Linear.weight layout
        qweight = qweight_t.transpose(0, 1).contiguous()
        return qweight, qmeta, maxq

    # ================================================================== #
    #                       Existing 4Bit-Forge API                      #
    # ================================================================== #

    @torch.no_grad()
    def build_quant_grid(
        self,
        weight: torch.Tensor,      # (C, R) transposed weight matrix
        group_size: int,
        bits: int,
        symmetric: bool = False,
        mode: str = "mse",      # "absmax" or "mse"
        quant_max_shrink: float = 0.2,
        quant_n_grid: int = 100,
        quant_norm: float = 2.4,
        impl: Literal["cuda", "triton"] = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Build groupwise quantization metadata in packed qmeta4 format.

        qmeta: (C, G, 4) uint8, where G = ceil(R / group_size)
        """
        if weight.ndim != 2:
            raise ValueError("build_quant_grid expects weight to be 2D (C, R).")
        C, R = int(weight.shape[0]), int(weight.shape[1])
        device = weight.device

        if group_size is None or group_size <= 0 or group_size > R:
            group_size = R
        if group_size % 32 != 0:
            raise ValueError(f"group_size must be a multiple of 32, got {group_size}")

        # Keep original dtype on CUDA, fp32 on CPU.
        W = weight if device.type == "cuda" else weight.to(torch.float32)
        if not W.is_contiguous():
            # Pre-solver path already makes W contiguous; this is just a safety net.
            W = W.contiguous()

        num_groups = (R + group_size - 1) // group_size
        padded_R = num_groups * group_size
        pad = padded_R - R

        if pad == 0:
            x_groups = W.view(C, num_groups, group_size).view(-1, group_size)
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
        else:
            full_R = (R // group_size) * group_size
            full_groups = full_R // group_size

            qmeta_parts = []
            maxq = None

            if full_groups > 0:
                x_full = W[:, :full_R].view(C, full_groups, group_size).reshape(-1, group_size)
                qmeta_full, maxq = self._build_quant_grid_groups(
                    x_groups=x_full,
                    bits=bits,
                    symmetric=symmetric,
                    mode=mode,
                    quant_max_shrink=quant_max_shrink,
                    quant_n_grid=quant_n_grid,
                    quant_norm=quant_norm,
                    impl=impl,
                )
                qmeta_parts.append(qmeta_full)

            tail_len = R - full_R
            x_tail = torch.zeros((C, group_size), device=device, dtype=W.dtype)
            if tail_len > 0:
                x_tail[:, :tail_len].copy_(W[:, full_R:R])
            x_tail = x_tail.view(-1, group_size)

            qmeta_tail, maxq2 = self._build_quant_grid_groups(
                x_groups=x_tail,
                bits=bits,
                symmetric=symmetric,
                mode=mode,
                quant_max_shrink=quant_max_shrink,
                quant_n_grid=quant_n_grid,
                quant_norm=quant_norm,
                impl=impl,
            )
            qmeta_parts.append(qmeta_tail)

            if maxq is None:
                maxq = maxq2

            qmeta_flat = torch.cat(qmeta_parts, dim=0)

        qmeta = qmeta_flat.view(C, num_groups, 4)
        return qmeta, maxq, pad

    @torch.no_grad()
    def solver(
        self,
        weight: torch.Tensor,        # (C, R), transposed weight, modified in-place by solver
        hessian_inv: torch.Tensor,   # (C, C), GPTQ: chol(H^{-1}); Babai: chol(H)
        qmeta: torch.Tensor,         # (C, G, 4) uint8
        maxq: torch.Tensor | None,
        group_size: int,
        bits: int,
    ) -> torch.Tensor:
        """
        CUDA fast-path + CPU reference fallback (GPTQ only).
        """
        if weight.device.type == "cuda":
            if self.algorithm == "babai":
                return cuda_kernels.babai_solver(
                    weight,
                    hessian_inv,
                    qmeta,
                    group_size,
                    bits,
                    self.block_size,
                )
            return cuda_kernels.gptq_solver(
                weight.float(),
                hessian_inv,
                qmeta,
                group_size,
                bits,
                self.block_size,
            )

        if self.algorithm == "babai":
            raise RuntimeError("Babai solver is CUDA-only (no CPU reference path).")

        # ---------------- CPU reference (kept for correctness/debug) ----------------
        assert weight.ndim == 2
        C, R = weight.shape
        assert hessian_inv.shape == (C, C)
        assert qmeta.ndim == 3 and qmeta.size(0) == C and qmeta.size(2) == 4
        assert group_size > 0

        device = weight.device
        w_dtype = weight.dtype

        num_groups = qmeta.size(1)
        maxq_bits = (1 << bits) - 1
        maxq_val = float(maxq_bits)

        W = weight.to(torch.float32).contiguous()
        Hcho = hessian_inv.to(torch.float32).contiguous()
        qweight = torch.empty_like(weight, dtype=torch.uint8, device=device)

        INV256 = 1.0 / 256.0
        block_size = 32

        for block_start in range(0, C, block_size):
            block_end = min(block_start + block_size, C)
            B = block_end - block_start

            delta_block = torch.zeros((B, R), dtype=torch.float32, device=device)

            for row_offset, j in enumerate(range(block_start, block_end)):
                row_meta = qmeta[j].to(torch.int32)

                lo = row_meta[:, 0]
                hi = row_meta[:, 1]
                log2_q88 = lo | (hi << 8)
                log2_q88 = torch.where(log2_q88 >= 32768, log2_q88 - 65536, log2_q88)

                log2_scale = log2_q88.float() * INV256
                scales = torch.exp2(log2_scale)
                inv_scales = torch.exp2(-log2_scale)

                qzeros_u8 = row_meta[:, 2]
                flags = row_meta[:, 3]

                is_sym = (flags & 1) != 0
                sym_q0 = (maxq_val + 1.0) * 0.5
                qzeros = torch.where(is_sym, sym_q0, qzeros_u8.float())

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

            H_block = Hcho[block_start:block_end, block_start:block_end]
            A_lower = H_block.t().contiguous()

            Delta_T = delta_block.t().contiguous()
            E_T = torch.empty_like(Delta_T)

            for r in range(R):
                b = Delta_T[r]
                x_vec = torch.empty((B,), dtype=torch.float32, device=device)
                for i in range(B):
                    s = 0.0 if i == 0 else torch.dot(A_lower[i, :i], x_vec[:i])
                    diag = A_lower[i, i]
                    x_vec[i] = (b[i] - s) / diag
                E_T[r] = x_vec

            E_J = E_T.t().contiguous()

            H_cross = Hcho[block_start:block_end, block_end:C]
            if H_cross.numel() > 0:
                W_tail = W[block_end:C, :]
                W_tail = W_tail - H_cross.t().mm(E_J)
                W[block_end:C, :] = W_tail

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
        impl: Literal["cuda", "triton"] = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x_groups.ndim != 2:
            raise ValueError("_build_quant_grid_groups expects (G_total, group_size) input.")
        device = x_groups.device
        mode_l = mode.lower()

        # ---------------------- CUDA path ----------------------
        if x_groups.device.type == "cuda":
            backend = cuda_kernels if impl == "cuda" else triton_kernels

            qmeta_bytes, maxq = backend.build_group_meta_packed(
                x_groups,
                bits,
                symmetric,
            )

            if mode_l == "mse":
                p = torch.linspace(
                    1.0,
                    quant_max_shrink,
                    quant_n_grid,
                    dtype=torch.float32,
                    device=device,
                )
                qmeta_bytes = backend.mse_scale_groups_packed(
                    x_groups,
                    p,
                    qmeta_bytes,
                    float(maxq.item()),
                    float(quant_norm),
                )

            return qmeta_bytes, maxq

        # ---------------------- CPU reference path ----------------------
        x = x_groups.to(torch.float32)
        G, group_size = x.shape

        maxq_val = (1 << bits) - 1
        maxq = torch.tensor(maxq_val, dtype=torch.float32, device=device)

        xmin = x.min(dim=-1).values
        xmax = x.max(dim=-1).values

        eps = 1e-12
        if symmetric:
            amax = torch.maximum(xmin.abs(), xmax.abs())
            scale = (2.0 / maxq) * amax + eps
            qzero = torch.full_like(scale, (maxq_val + 1.0) * 0.5)
        else:
            scale = (xmax - xmin) / maxq + eps
            qzero = torch.round(-xmin / scale).clamp_(0.0, float(maxq_val))

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

            # (CPU ref only; ok to keep simple)
            for g in range(G):
                xg = x[g]
                base_s = float(scale[g].item())
                q0 = float(qzero[g].item())

                best_loss = float("inf")
                best_s = base_s

                for k in range(p.numel()):
                    s = base_s * float(p[k].item())
                    if s <= 0.0:
                        continue
                    q = torch.round(xg / s + q0).clamp_(0.0, maxq_f)
                    y = (q - q0) * s
                    diff = (y - xg).abs()
                    loss = diff.pow(quant_norm).sum().item()
                    if loss < best_loss:
                        best_loss = loss
                        best_s = s

                new_scale[g] = best_s

            scale = new_scale

        qmeta_bytes = self._encode_qmeta_groups(scale, qzero, symmetric=symmetric)
        return qmeta_bytes, maxq

    # ------------------------------------------------------------------ #
    # qmeta encode/decode helpers
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _encode_qmeta_groups(
        self,
        scale_g: torch.Tensor,   # (G,) float32
        qzero_g: torch.Tensor,   # (G,) float32
        symmetric: bool = False,
    ) -> torch.Tensor:
        if scale_g.ndim != 1 or qzero_g.ndim != 1 or scale_g.shape != qzero_g.shape:
            raise ValueError("_encode_qmeta_groups expects 1D tensors with same shape.")
        device = scale_g.device

        eps = 1e-12
        s = torch.clamp(scale_g, min=eps)
        log2_fp = torch.log2(s)
        log2_q88 = torch.round(log2_fp * 256.0).to(torch.int16)

        lo = (log2_q88 & 0xFF).to(torch.uint8)
        hi = ((log2_q88 >> 8) & 0xFF).to(torch.uint8)

        qzero_u8 = qzero_g.round().clamp(0, 255).to(torch.uint8)

        qmeta = torch.empty((scale_g.shape[0], 4), dtype=torch.uint8, device=device)
        qmeta[:, 0] = lo
        qmeta[:, 1] = hi
        qmeta[:, 2] = qzero_u8
        qmeta[:, 3] = 1 if symmetric else 0  # bit0 = symmetric (optional; decode paths can ignore)
        return qmeta

    @torch.no_grad()
    def _decode_qmeta_groups(
        self,
        qmeta_bytes: torch.Tensor,         # (G, 4) uint8
        out_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if qmeta_bytes.ndim != 2 or qmeta_bytes.size(1) != 4 or qmeta_bytes.dtype != torch.uint8:
            raise ValueError("_decode_qmeta_groups expects (G,4) uint8.")

        lo = qmeta_bytes[:, 0].to(torch.int16)
        hi = qmeta_bytes[:, 1].to(torch.int16)
        log2_q88 = (lo | (hi << 8)).to(torch.int16)
        log2_fp = log2_q88.to(torch.float32) / 256.0
        scale = torch.exp2(log2_fp)

        qzero = qmeta_bytes[:, 2].to(torch.float32)

        if out_dtype is not None:
            scale = scale.to(dtype=out_dtype)
            qzero = qzero.to(dtype=out_dtype)

        return scale.to(device=qmeta_bytes.device), qzero.to(device=qmeta_bytes.device)
