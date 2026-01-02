import os
import json
import shutil
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set, Union, Sequence
from collections import defaultdict, OrderedDict, deque

import torch
import torch.nn as nn
from safetensors import safe_open 
from huggingface_hub import hf_hub_download
from typing import List
import torch
import torch.nn as nn
from accelerate.utils import set_module_tensor_to_device


# -----------------------------
# Small utilities
# -----------------------------

def _tensor_bytes(t: torch.Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def _free_bytes(path: str) -> int:
    # path must exist; we create tmp_dir before use.
    return int(shutil.disk_usage(path).free)


def _is_local_dir(repo_id: str) -> bool:
    return os.path.isdir(repo_id)


def _now() -> float:
    return time.time()


class ParamSliceProxy:
    def __init__(self, parent: torch.nn.Module, param_name: str, expert_id: int, *, transpose: bool = True):
        self.parent = parent
        self.param_name = param_name
        self.expert_id = expert_id
        self.transpose = transpose

    @property
    def weight(self) -> torch.Tensor:
        p = getattr(self.parent, self.param_name)      # e.g. [E, in, out]
        w = p[self.expert_id]                          # [in, out]
        return w.transpose(0, 1) if self.transpose else w   # => [out, in] for GPTQ

    @property
    def dtype(self): return self.weight.dtype

    @property
    def device(self): return self.weight.device

def list_layers(block: nn.Module) -> Dict[str, Union[nn.Linear, ParamSliceProxy]]:
    layers: Dict[str, Union[nn.Linear, ParamSliceProxy]] = {}

    # Normal path: nn.Linear modules (DeepSeek-style experts are nn.Linear)
    detect_experts = False
    for n, m in block.named_modules():
        if isinstance(m, nn.Linear):
            layers[n] = m
        
    """
    # GPT-OSS path: fused expert weights as Parameters on block.mlp.experts
    mlp = getattr(block, "mlp", None)
    experts = getattr(mlp, "experts", None) if mlp is not None else None
    
    if experts is not None:
        gu = getattr(experts, "gate_up_proj", None)   # [E,H,2D]
        dn = getattr(experts, "down_proj", None)      # [E,D,H]

        if isinstance(gu, nn.Parameter) and gu.ndim == 3 and isinstance(dn, nn.Parameter) and dn.ndim == 3:
            E = gu.shape[0]
            for e in range(E):
                layers[f"mlp.experts.{e}.gate_up_proj"] = ParamSliceProxy(experts, "gate_up_proj", e)
                layers[f"mlp.experts.{e}.down_proj"]    = ParamSliceProxy(experts, "down_proj", e)
    """

    return layers
# -----------------------------
# set_module_tensor_to_device (fallback)
# -----------------------------
def set_module_tensor_to_device(module: nn.Module, tensor_name: str, device: str, value: torch.Tensor) -> None:
    """
    Minimal replacement for accelerate.utils.set_module_tensor_to_device.
    Supports setting parameters and buffers by attribute name.
    """
    # If it's a Parameter, keep as Parameter; else buffer/tensor attribute.
    if tensor_name in getattr(module, "_parameters", {}):
        req_grad = module._parameters[tensor_name].requires_grad if module._parameters[tensor_name] is not None else False
        module._parameters[tensor_name] = nn.Parameter(value.to(device=device), requires_grad=req_grad)
        return

    if tensor_name in getattr(module, "_buffers", {}):
        module._buffers[tensor_name] = value.to(device=device)
        return

    # Fallback: attribute assignment
    current = getattr(module, tensor_name, None)
    if isinstance(current, nn.Parameter):
        setattr(module, tensor_name, nn.Parameter(value.to(device=device), requires_grad=current.requires_grad))
    else:
        setattr(module, tensor_name, value.to(device=device))


# -----------------------------
# RAM-side shard tensor cache (optional)
# -----------------------------

class ShardLRU:
    """
    Cache extracted tensors (by shard) in CPU RAM to reduce repeated safetensors reads.

    Stores: shard_name -> (tensors_dict, total_bytes, last_touch)
    Eviction: LRU by access, bounded by max_bytes.
    """
    def __init__(self, max_bytes: int = 0):
        self.max_bytes = int(max_bytes)
        self._od: "OrderedDict[str, Tuple[Dict[str, torch.Tensor], int, float]]" = OrderedDict()
        self._bytes: int = 0

    def enabled(self) -> bool:
        return self.max_bytes > 0

    def get(self, shard_name: str) -> Optional[Dict[str, torch.Tensor]]:
        if not self.enabled():
            return None
        if shard_name not in self._od:
            return None
        tensors, nbytes, _ = self._od.pop(shard_name)
        self._od[shard_name] = (tensors, nbytes, _now())
        return tensors

    def put(self, shard_name: str, tensors: Dict[str, torch.Tensor], total_bytes: int) -> None:
        if not self.enabled():
            return
        total_bytes = int(total_bytes)

        # If already present, replace
        if shard_name in self._od:
            _, old_bytes, _ = self._od.pop(shard_name)
            self._bytes -= old_bytes

        self._od[shard_name] = (tensors, total_bytes, _now())
        self._bytes += total_bytes

        # Evict to budget
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        if not self.enabled():
            return
        while self._bytes > self.max_bytes and self._od:
            shard, (tensors, nbytes, _) = self._od.popitem(last=False)
            # Drop refs
            try:
                tensors.clear()
            except Exception:
                pass
            self._bytes -= nbytes

    def clear(self) -> None:
        self._od.clear()
        self._bytes = 0


# -----------------------------
# Disk-side shard window (reserve-aware)
# -----------------------------

@dataclass
class DiskWindowConfig:
    """
    shard_bytes:
      - If >0: dynamic window size based on free disk and reserve.
      - If 0: use max_shards as a fixed-count FIFO window.
    """
    shard_bytes: int = 0                 # e.g., int(5.36 * 1024**3)
    safety_bytes: int = int(2 * 1024**3) # extra safety margin on disk
    max_shards: int = 8                  # used only when shard_bytes == 0
    use_lru_touch: bool = True           # move reused shard to MRU


class ShardDiskWindow:
    """
    Maintains a window of shard files inside tmp_dir.

    - If cfg.shard_bytes > 0: computes cap_shards dynamically based on:
        cap = floor((free - reserve - safety) / shard_bytes), min 1
      and trims the window to that cap.
    - Else: maintains a fixed max_shards window.

    Eviction policy:
      - MRU tracked by deque; we evict from the oldest end.
      - Pinned paths are never evicted; if all are pinned, raises.
    """
    def __init__(self, tmp_dir: str, cfg: DiskWindowConfig):
        self.tmp_dir = tmp_dir
        self.cfg = cfg
        os.makedirs(self.tmp_dir, exist_ok=True)
        self._window: "deque[str]" = deque()  # stores absolute local paths (within tmp_dir or symlink targets)

    def _cap(self, reserve_bytes: int) -> int:
        if self.cfg.shard_bytes > 0:
            free = _free_bytes(self.tmp_dir)
            usable = free - int(reserve_bytes) - int(self.cfg.safety_bytes)
            if usable <= 0:
                return 1
            return max(1, int(usable // int(self.cfg.shard_bytes)))
        return max(1, int(self.cfg.max_shards))

    def _touch(self, path: str) -> None:
        if not self.cfg.use_lru_touch:
            if path not in self._window:
                self._window.append(path)
            return
        # LRU-ish: move to MRU
        try:
            self._window.remove(path)
        except ValueError:
            pass
        self._window.append(path)

    def _evict_one(self, pinned_paths: Set[str]) -> bool:
        # Evict one oldest non-pinned shard file
        for _ in range(len(self._window)):
            cand = self._window.popleft()
            if cand in pinned_paths:
                self._window.append(cand)
                continue
            # Delete file if exists (best-effort)
            try:
                os.remove(cand)
            except FileNotFoundError:
                pass
            except Exception:
                pass
            return True
        return False

    def _trim_to_cap(self, cap: int, pinned_paths: Set[str]) -> None:
        while len(self._window) > cap:
            if not self._evict_one(pinned_paths):
                break

    def ensure_space_for_download(
        self,
        *,
        reserve_bytes: int,
        pinned_paths: Set[str],
    ) -> None:
        """
        Ensure there is enough free disk for:
          reserve + safety + (one shard download, if shard_bytes known; else 0)
        """
        shard_need = int(self.cfg.shard_bytes) if self.cfg.shard_bytes > 0 else 0
        need_free = int(reserve_bytes) + int(self.cfg.safety_bytes) + shard_need
        while _free_bytes(self.tmp_dir) < need_free:
            if not self._evict_one(pinned_paths):
                raise RuntimeError(
                    "Disk pressure: cannot free enough space (only pinned shards remain). "
                    "Increase disk or reduce reserve/lookahead."
                )


    def download_shard(
        self,
        repo_id: str,
        shard_filename: str,
        *,
        reserve_bytes: int = 0,
        pinned_filenames: Optional[Set[str]] = None,
    ) -> str:
        """
        Returns local shard path. For remote repos, uses hf_hub_download into tmp_dir.
        For local repos, tries to create a symlink into tmp_dir (or returns source path).
        Applies eviction policy to keep window bounded.
        """
        if pinned_filenames is None:
            pinned_filenames = set()
        pinned_paths: Set[str] = {os.path.join(self.tmp_dir, s) for s in pinned_filenames}

        local_path = os.path.join(self.tmp_dir, shard_filename)

        # If already present, touch + trim and return
        if os.path.exists(local_path):
            self._touch(local_path)
            cap = self._cap(reserve_bytes)
            self._trim_to_cap(cap, pinned_paths)
            return local_path

        # Make space before fetching
        self.ensure_space_for_download(reserve_bytes=reserve_bytes, pinned_paths=pinned_paths)

        # Local directory case: try to symlink/hardlink into tmp_dir
        if _is_local_dir(repo_id):
            src = os.path.join(repo_id, shard_filename)
            if not os.path.exists(src):
                raise FileNotFoundError(f"Local shard not found: {src}")

            # Create symlink for bookkeeping if possible
            try:
                os.symlink(src, local_path)
            except FileExistsError:
                pass
            except Exception:
                # Fallback: just use src directly (cannot reclaim disk by deleting src)
                local_path = src

            self._touch(local_path)
            cap = self._cap(reserve_bytes)
            self._trim_to_cap(cap, pinned_paths)
            return local_path


        path = hf_hub_download(
            repo_id=repo_id,
            filename=shard_filename,
            repo_type="model",
            local_dir=self.tmp_dir,
            force_download=False,
        )

        # Track and trim
        self._touch(path)
        cap = self._cap(reserve_bytes)
        self._trim_to_cap(cap, pinned_paths)
        return path


# -----------------------------
# Index loading
# -----------------------------

def load_safetensors_index(model_name_or_path: str, tmp_dir: str = "/tmp/hf_jit") -> Dict[str, Any]:
    """
    Loads the safetensors index JSON (weight_map) for a sharded model.

    Tries, in order:
      1) model.safetensors.index.json
      2) pytorch_model.bin.index.json

    Returns the parsed JSON dict (must contain "weight_map").
    """
    os.makedirs(tmp_dir, exist_ok=True)

    candidates = ["model.safetensors.index.json", "pytorch_model.bin.index.json"]

    # Local dir: read directly if present
    if _is_local_dir(model_name_or_path):
        for fname in candidates:
            p = os.path.join(model_name_or_path, fname)
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "weight_map" not in data:
                    raise KeyError(f"Index missing weight_map: {p}")
                return data
        raise FileNotFoundError(f"No index file found in local dir: {model_name_or_path}")

    # Remote HF repo id: download index
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is not available to download index from remote repo.")

    last_err: Optional[Exception] = None
    for fname in candidates:
        try:
            idx_path = hf_hub_download(
                repo_id=model_name_or_path,
                filename=fname,
                repo_type="model",
                local_dir=tmp_dir,
                force_download=False,
            )
            with open(idx_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "weight_map" not in data:
                raise KeyError(f"Index missing weight_map: {fname}")
            return data
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed to load index for {model_name_or_path}: {last_err}")


# -----------------------------
# Tensor injection / metaization
# -----------------------------
def _walk_module_path(root: nn.Module, parts: list[str]) -> nn.Module:
    mod: Any = root
    for p in parts:
        if p.isdigit():
            mod = mod[int(p)]  # ModuleList / list-like
        else:
            mod = getattr(mod, p)
    return mod


def inject_tensor_by_name(model: nn.Module, name: str, tensor: torch.Tensor, device: str = "cpu") -> None:
    """
    Inject a tensor into model by dotted name. Handles ModuleList indices.
    """
    parts = name.split(".")
    attr = parts[-1]
    module_path = parts[:-1]

    try:
        mod = _walk_module_path(model, module_path) if module_path else model
    except Exception:
        return

    # If already registered, overwrite
    if hasattr(mod, attr) or attr in getattr(mod, "_parameters", {}) or attr in getattr(mod, "_buffers", {}):
        set_module_tensor_to_device(mod, attr, device=device, value=tensor)
        return

    # Register placeholder then set
    if attr in ("weight", "bias") or attr in getattr(mod, "_parameters", {}):
        placeholder = nn.Parameter(
            torch.empty(tensor.shape, device="meta", dtype=tensor.dtype),
            requires_grad=False,
        )
        mod.register_parameter(attr, placeholder)
    else:
        placeholder = torch.empty(tensor.shape, device="meta", dtype=tensor.dtype)
        mod.register_buffer(attr, placeholder, persistent=True)

    set_module_tensor_to_device(mod, attr, device=device, value=tensor)


def metaize_module_(module: nn.Module) -> None:
    """Move params/buffers back to META to free CPU RAM."""
    for n, p in list(module.named_parameters(recurse=True)):
        if p is None:
            continue
        meta = torch.empty(p.shape, device="meta", dtype=p.dtype)
        try:
            parts = n.split(".")
            attr = parts[-1]
            mod = _walk_module_path(module, parts[:-1]) if len(parts) > 1 else module
            set_module_tensor_to_device(mod, attr, device="meta", value=meta)
        except Exception:
            pass

    for n, b in list(module.named_buffers(recurse=True)):
        if b is None:
            continue
        meta = torch.empty(b.shape, device="meta", dtype=b.dtype)
        try:
            parts = n.split(".")
            attr = parts[-1]
            mod = _walk_module_path(module, parts[:-1]) if len(parts) > 1 else module
            set_module_tensor_to_device(mod, attr, device="meta", value=meta)
        except Exception:
            pass


# -----------------------------
# Main: prefix streaming loader
# -----------------------------

def _get_submodule(root: nn.Module, path: str) -> nn.Module:
    return _walk_module_path(root, path.split(".")) if path else root



def assign_param_(layer: nn.Module, name: str, value: torch.Tensor):
    """Safe assignment for meta/uninitialized params created under init_empty_weights()."""
    assert hasattr(layer, name)
    p = getattr(layer, name)
    v = value.detach()

    # meta / uninitialized → replace the Parameter
    if getattr(p, "is_meta", False) or (hasattr(p, "device") and p.device.type == "meta"):
        setattr(layer, name, nn.Parameter(v, requires_grad=False))
        return

    # normal tensor → copy into existing storage
    p.copy_(v)


# Keys used only for materialization; don't inject these into the model.
_MATERIALIZE_SIDECAR_SUFFIXES = (
    ".weight_scale", ".weight_scales", ".weight_scale_inv", ".weight_zero",
    ".qweight", ".qzeros", ".scales", ".g_idx", ".perm",
    "_blocks", "_scales",   # NOTE: do NOT include "_bias"
)



def _fp8_dtypes() -> set[torch.dtype]:
    out = set()
    for name in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz"):
        if hasattr(torch, name):
            out.add(getattr(torch, name))
    return out

_FP8_DTYPES = _fp8_dtypes()

# MXFP4 mag table (E2M1) used by gpt-oss MXFP4
_MXFP4_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

def _e8m0_to_scale_fp32(e_u8: torch.Tensor) -> torch.Tensor:
    e = e_u8.to(torch.int16)
    return torch.where(
        e == 0,
        torch.tensor(2.0 ** -23, dtype=torch.float32),
        torch.pow(torch.tensor(2.0, dtype=torch.float32), (e.to(torch.float32) - 127.0)),
    )

def _dequant_mxfp4_blocks_to_fp(blocks_u8: torch.Tensor, scales_u8: torch.Tensor) -> torch.Tensor:
    # blocks: [..., NB, 16] bytes -> 32 fp4 codes
    b = blocks_u8.to(torch.uint8)
    lo = b & 0x0F
    hi = (b >> 4) & 0x0F
    codes = torch.stack([lo, hi], dim=-1).reshape(*b.shape[:-1], 32)  # [..., NB, 32]

    mag = (codes & 0x7).to(torch.long)
    sign = ((codes & 0x8) != 0)
    vals = _MXFP4_MAG[mag]
    vals = torch.where(sign, -vals, vals)

    scale = _e8m0_to_scale_fp32(scales_u8).unsqueeze(-1)  # [..., NB, 1]
    out = (vals * scale).reshape(*vals.shape[:-2], vals.shape[-2] * vals.shape[-1])  # [..., NB*32]
    return out.transpose(-2, -1)  # fp32

def _apply_block_scale_inv_2d(w_fp32: torch.Tensor, s_inv_2d: torch.Tensor) -> torch.Tensor:
    # w: [O, I], s_inv: [O/128, I/128]
    O, I = w_fp32.shape
    so, si = s_inv_2d.shape
    assert O % so == 0 and I % si == 0, (w_fp32.shape, s_inv_2d.shape)

    br = O // so
    bc = I // si
    w4 = w_fp32.reshape(so, br, si, bc)
    s = s_inv_2d.to(torch.float32)
    w4 = w4 * (1.0 / s).reshape(so, 1, si, 1)
    return w4.reshape(O, I)


def _apply_block_scale(
    w: torch.Tensor,
    s: torch.Tensor,
    weight_shape: Union[Sequence[int], torch.Tensor],
    *,
    bits: int = 4,
    symmetric: bool = True,
    scale_is_inv: bool = False,   # set True if the tensor is weight_scale_inv
) -> torch.Tensor:
    """
    Kimi-style: int4 packed into int32 + per-(group_size) scale.

    Inputs
      w: packed int32 (or already-dense float/int tensor)
      s: scales tensor (per group)
      weight_shape: [O, I] (dense target)

    Returns
      dense fp32 weight of shape [O, I]
    """
    if torch.is_tensor(weight_shape):
        weight_shape = [int(x) for x in weight_shape.flatten().tolist()]
    else:
        weight_shape = [int(x) for x in list(weight_shape)]
    assert len(weight_shape) == 2, f"Expected 2D weight_shape, got {weight_shape}"
    O, I = weight_shape
    dense_numel = O * I

    # scale handling
    sf = s.to(dtype=torch.float32, device=w.device)
    if scale_is_inv:
        sf = 1.0 / sf

    # infer group_size from counts (works for your numbers)
    if sf.numel() == dense_numel:
        group_size = 1
        sf2 = sf.view(O, I)
    else:
        # group_size = dense_numel / num_scales
        assert dense_numel % sf.numel() == 0, (
            f"Cannot infer group_size: dense_numel={dense_numel} not divisible by scales.numel={sf.numel()}"
        )
        group_size = dense_numel // sf.numel()

        # expect scales shaped [O, I/group_size] (or broadcastable to it)
        I_groups = I // group_size
        if sf.numel() == O * I_groups:
            sf2 = sf.view(O, I_groups).repeat_interleave(group_size, dim=1)
        else:
            # last-resort broadcast: try to reshape to [O, I_groups]
            sf2 = sf.reshape(O, I_groups).repeat_interleave(group_size, dim=1)

    # If already dense, just apply scales
    if w.numel() == dense_numel:
        w_fp = w.to(dtype=torch.float32).view(O, I)
        return w_fp * sf2

    # Packed int32 path: infer values_per_word
    assert dense_numel % w.numel() == 0, (
        f"Packed weight numel doesn't divide dense numel: dense={dense_numel}, packed={w.numel()}"
    )
    vals_per = dense_numel // w.numel()
    # For int32 packing, vals_per should be 32/bits (=8 when bits=4)
    expected = 32 // bits
    assert vals_per == expected, f"Unexpected packing: vals_per={vals_per}, expected={expected} (bits={bits})"

    # reshape packed to [O, I/vals_per]
    assert I % vals_per == 0, f"I={I} not divisible by vals_per={vals_per}"
    I_packed = I // vals_per
    wp = w.to(device=w.device)
    wp = wp.view(O, I_packed).to(torch.int32)

    mask = (1 << bits) - 1
    # unpack to uint4 values in [0..15]
    # shape: [O, I_packed, vals_per] -> [O, I]
    parts = []
    for i in range(vals_per):
        parts.append(((wp >> (i * bits)) & mask).to(torch.int16))
    u = torch.stack(parts, dim=-1).reshape(O, I)  # uint4

    if symmetric:
        # map uint4 -> signed int4 via two's complement
        sign_bit = 1 << (bits - 1)  # 8
        u = torch.where(u >= sign_bit, u - (1 << bits), u)  # [-8..7]

    w_fp = u.to(torch.float32) * sf2
    return w_fp

def _maybe_materialize_gptoss_expert_params(block: nn.Module, state_tensors: dict[str, torch.Tensor], dtype: torch.dtype) -> None:
    # gpt-oss pattern: mlp.experts.{gate_up_proj,down_proj}_{blocks,scales} -> mlp.experts.{gate_up_proj,down_proj}
    for proj in ("gate_up_proj", "down_proj"):
        bk = f"mlp.experts.{proj}_blocks"
        sk = f"mlp.experts.{proj}_scales"
        if bk not in state_tensors or sk not in state_tensors:
            continue

        try:
            experts_mod = _walk_module_path(block, ["mlp", "experts"])
        except Exception:
            continue

        w_fp32 = _dequant_mxfp4_blocks_to_fp(state_tensors[bk], state_tensors[sk])  # fp32
        w = w_fp32.to(dtype=dtype, device="cpu")

        # IMPORTANT: set as a PARAM on experts_mod (proj is the attribute name)
        set_module_tensor_to_device(experts_mod, proj, device="cpu", value=w)

        # consume sidecars so we never inject them
        state_tensors.pop(bk, None)
        state_tensors.pop(sk, None)

def materialize_block_weights_to_fp(
    block: nn.Module,
    state_tensors: dict[str, torch.Tensor],
    *,
    group_size: int,
    bits: int,
    dtype: torch.dtype,
    list_layers_fn=None,
) -> None:
    # gpt-oss fused expert params (if present in this checkpoint)
    _maybe_materialize_gptoss_expert_params(block, state_tensors, dtype)

    if list_layers_fn is None:
        list_layers_fn = list_layers

    layers = list_layers_fn(block)
    if isinstance(layers, dict):
        items = layers.items()
    else:
        items = layers  # assume iterable of (name, layer)

    for lname, layer in items:
        w_key = f"{lname}.weight"
        if w_key not in state_tensors:
            w_key = f"{lname}.weight_packed"
            if w_key not in state_tensors:
                    continue

        w_raw = state_tensors[w_key]

        # DeepSeek FP8: float8 weight + weight_scale_inv (2D blockwise)
        if w_raw.dtype in _FP8_DTYPES:
            w_fp32 = w_raw.to(torch.float32)
            inv_key = f"{lname}.weight_scale_inv"
            if inv_key in state_tensors and state_tensors[inv_key].ndim == 2 and w_fp32.ndim == 2:
                w_fp32 = _apply_block_scale_inv_2d(w_fp32, state_tensors[inv_key])
                state_tensors.pop(inv_key, None)

            w = w_fp32.to(dtype=dtype, device="cpu")
            set_module_tensor_to_device(layer, "weight", device="cpu", value=w)
            state_tensors.pop(w_key, None)
            continue

        # Fallback: int + scale-ish (optional)
        w_fp32 = w_raw.to(torch.float32)
        for scale_key in (f"{lname}.weight_scales", f"{lname}.weight_scale"):
            if scale_key in state_tensors:
                s = state_tensors[scale_key].to(torch.float32)
                shape_key = f"{lname}.weight_shape"
                if shape_key in state_tensors:
                    _shape = state_tensors[shape_key]
                    try:
                        w_fp32 = _apply_block_scale(w_fp32, s, _shape.detach().tolist(), symmetric=True)
                    except:
                        w_fp32 = _apply_block_scale(w_fp32, s, _shape.detach().tolist(), symmetric=False)

                else:
                    w_fp32 = w_fp32 * s
                state_tensors.pop(scale_key, None)
        
        inv_key = f"{lname}.weight_scale_inv"
        if inv_key in state_tensors:
            s = state_tensors[inv_key].to(torch.float32)
            shape_key = f"{lname}.weight_shape"
            if shape_key in state_tensors:
                    _shape = state_tensors[shape_key]
                    try:
                        w_fp32 = _apply_block_scale(w_fp32, s, _shape.detach().tolist(), symmetric=True, scale_is_inv=True)
                    except:
                        w_fp32 = _apply_block_scale(w_fp32, s, _shape.detach().tolist(), symmetric=False, scale_is_inv=True)
            else:
                w_fp32 = w_fp32 * (1.0 / s)
            state_tensors.pop(inv_key, None)

        if w_raw.dtype in (torch.float16, torch.bfloat16, torch.float32):
            w = w_raw.to(dtype=dtype, device="cpu")
            set_module_tensor_to_device(layer, "weight", device="cpu", value=w)
            state_tensors.pop(w_key, None)
            continue

        w = w_fp32.to(dtype=dtype, device="cpu")
        set_module_tensor_to_device(layer, "weight", device="cpu", value=w)
        state_tensors.pop(w_key, None)






def jit_load_prefix_to_cpu(
    model: nn.Module,
    repo_id: str,
    weight_map: dict,
    prefixes: list[str],
    tmp_dir: str,
    lru,  # your ShardLRU
    *,
    reserve_bytes: int = 0,
    disk_window=None,  # optional: your reserve-aware window; else uses _download_shard

    # Requant/materialize hook (optional)
    materialize_block: nn.Module | None = None,
    materialize_prefix: str | None = None,     # e.g. f"model.layers.{i}."
    materialize_fn=None,                       # your materialize_block_weights_to_fp
    group_size: int | None = None,
    bits: int | None = None,
    dtype: torch.dtype | None = None,
    list_layers_fn=None,                       # engine.list_layers
) -> tuple[str, int]:
    """
    Loads tensors for given prefixes into CPU model.

    If materialize_block is provided:
      1) collect tensors
      2) call materialize_fn(materialize_block, prefix_relative_state, ...)
      3) inject only non-weight / non-sidecar tensors
    """

    # 1) gather needed tensor names
    needed = []
    for name in weight_map.keys():
        for pfx in prefixes:
            if name.startswith(pfx):
                needed.append(name)
                break
    if not needed:
        return "0", 0

    by_shard = defaultdict(list)
    for name in needed:
        by_shard[weight_map[name]].append(name)

    # 2) if materializing, precompute which exact ".weight" names belong to linears in this block
    linear_weight_fullnames = set()
    if materialize_block is not None:
        assert materialize_prefix is not None
        assert materialize_fn is not None
        assert dtype is not None
        assert list_layers_fn is not None

        layers = list_layers_fn(materialize_block)  # {lname: nn.Linear}
        for lname in layers.keys():
            linear_weight_fullnames.add(f"{materialize_prefix}{lname}.weight")

    # 3) collect tensors (no injection yet)
    collected: dict[str, torch.Tensor] = {}
    pinned_filenames = set(by_shard.keys())

    for shard_name, names in by_shard.items():
        tensors = lru.get(shard_name) or {}
        missing = [tname for tname in names if tname not in tensors]

        if missing:
            shard_path = disk_window.download_shard(
                    repo_id, shard_name,
                    reserve_bytes=int(reserve_bytes),
                    pinned_filenames=pinned_filenames,
                )

            with safe_open(shard_path, framework="pt", device="cpu") as f:
                keys = set(f.keys())
                for tname in missing:
                    if tname not in keys:
                        raise KeyError(f"[JIT] tensor not found in shard: {tname} (shard={shard_name})")
                    tensors[tname] = f.get_tensor(tname)

            total_bytes = sum(_tensor_bytes(t) for t in tensors.values())
            lru.put(shard_name, tensors, total_bytes)

        for tname in names:
            collected[tname] = tensors[tname]

    # 4) materialize linears first (prefix-relative dict)
    if materialize_block is not None:
        pfx = materialize_prefix
        rel = {}
        for k, v in collected.items():
            if k.startswith(pfx):
                rel[k[len(pfx):]] = v  # strip "model.layers.i." prefix

        materialize_fn(
            materialize_block,
            rel,
            group_size=group_size,
            bits=bits,
            dtype=dtype,
            list_layers_fn=list_layers_fn
        )

    # 5) inject everything else (skip linear weights + sidecars)
    bytes_loaded = 0
    for tname, t in collected.items():
        if materialize_block is not None:
            if tname in linear_weight_fullnames:
                continue
            if tname.endswith(_MATERIALIZE_SIDECAR_SUFFIXES):
                continue

        inject_tensor_by_name(model, tname, t, device="cpu")
        bytes_loaded += _tensor_bytes(t)

    return str(len(by_shard)), bytes_loaded

# -----------------------------------------------------------------------------
# Optional: helper to precompute block->shards mapping from weight_map
# -----------------------------------------------------------------------------

def build_block_to_shards(weight_map: Dict[str, str]) -> Dict[int, Set[str]]:
    """
    Returns {block_id -> set(shard_filenames)} for names starting with "model.layers.{i}."
    Useful for pinning current+next block shard filenames during JIT streaming.
    """
    out: Dict[int, Set[str]] = defaultdict(set)
    for pname, shard in weight_map.items():
        if pname.startswith("model.layers."):
            parts = pname.split(".")
            if len(parts) >= 3 and parts[2].isdigit():
                bid = int(parts[2])
                out[bid].add(shard)
    return dict(out)
