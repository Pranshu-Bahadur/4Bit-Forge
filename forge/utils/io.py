from __future__ import annotations

import os
import json
import shutil
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, OrderedDict, deque

from . import engine

import torch
import torch.nn as nn
from safetensors import safe_open 
from huggingface_hub import hf_hub_download


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

def inject_tensor_by_name(model: nn.Module, name: str, tensor: torch.Tensor, device: str = "cpu") -> None:
    """
    Inject a tensor into model by dotted name.
    Supports parameters (weight/bias) and buffers.
    """
    parts = name.split(".")
    attr = parts[-1]
    module_path = ".".join(parts[:-1])

    mod = model
    if module_path:
        try:
            for p in module_path.split("."):
                mod = getattr(mod, p)
        except AttributeError:
            return  # silently ignore missing paths

    # If already registered, overwrite
    if hasattr(mod, attr) or attr in getattr(mod, "_parameters", {}) or attr in getattr(mod, "_buffers", {}):
        set_module_tensor_to_device(mod, attr, device=device, value=tensor)
        return

    # Register placeholder then set
    if attr in ("weight", "bias"):
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
    """
    Move params/buffers back to META to free CPU RAM after a block is done.
    """
    # Parameters
    for n, p in list(module.named_parameters(recurse=True)):
        if p is None:
            continue
        meta = torch.empty(p.shape, device="meta", dtype=p.dtype)
        try:
            parts = n.split(".")
            attr = parts[-1]
            mod = module
            for k in parts[:-1]:
                mod = getattr(mod, k)
            set_module_tensor_to_device(mod, attr, device="meta", value=meta)
        except Exception:
            pass

    # Buffers
    for n, b in list(module.named_buffers(recurse=True)):
        if b is None:
            continue
        meta = torch.empty(b.shape, device="meta", dtype=b.dtype)
        try:
            parts = n.split(".")
            attr = parts[-1]
            mod = module
            for k in parts[:-1]:
                mod = getattr(mod, k)
            set_module_tensor_to_device(mod, attr, device="meta", value=meta)
        except Exception:
            pass


# -----------------------------
# Main: prefix streaming loader
# -----------------------------

def _get_submodule(root: nn.Module, path: str) -> nn.Module:
    mod = root
    if path:
        for p in path.split("."):
            mod = getattr(mod, p)
    return mod


def materialize_block_weights_to_fp(block, state_tensors: dict, *, dtype):
    """
    block: the actual module with nn.Linear layers already created (meta -> real tensors once assigned)
    state_tensors: loaded tensors for this prefix from the source checkpoint
    goal: write float weights into block's nn.Linear.weight (CPU), so the block can run forward.
    """
    layers = engine.list_layers(block)  # your existing layer enumerator
    if not layers:
        return

    for lname, layer in layers.items():
        w_key = f"{lname}.weight"

        if w_key in state_tensors and state_tensors[w_key].dtype:
            scale_key = f"{lname}.weight_scales"
            zero_key  = f"{lname}.weight_zero"  # optional
            if scale_key in state_tensors:
                w_int = state_tensors[w_key].to(torch.float32)
                s = state_tensors[scale_key].to(torch.float32)
                if zero_key in state_tensors:
                    z = state_tensors[zero_key].to(torch.float32)
                    w_fp = (w_int - z) * s
                else:
                    w_fp = w_int * s
                layer.weight.data = w_fp.to(dtype=dtype, device="cpu")
                continue
        
        if w_key in state_tensors and state_tensors[w_key].dtype:
            scale_key = f"{lname}.weight_scale"
            zero_key  = f"{lname}.weight_zero"  # optional
            if scale_key in state_tensors:
                w_int = state_tensors[w_key].to(torch.float32)
                s = state_tensors[scale_key].to(torch.float32)
                if zero_key in state_tensors:
                    z = state_tensors[zero_key].to(torch.float32)
                    w_fp = (w_int - z) * s
                else:
                    w_fp = w_int * s
                layer.weight.data = w_fp.to(dtype=dtype, device="cpu")
                continue

        if w_key in state_tensors and state_tensors[w_key].dtype:
                scale_key = f"{lname}.weight_scale_inv"
                zero_key  = f"{lname}.weight_zero"  # optional
                if scale_key in state_tensors:
                    w_int = state_tensors[w_key].to(torch.float32)
                    s = state_tensors[scale_key].to(torch.float32)
                    if zero_key in state_tensors:
                        z = state_tensors[zero_key].to(torch.float32)
                        w_fp = (w_int - z) * (1/s)
                    else:
                        w_fp = w_int * (1/s)
                    layer.weight.data = w_fp.to(dtype=dtype, device="cpu")
                    continue

        if w_key in state_tensors and state_tensors[w_key].dtype in (torch.float16, torch.bfloat16, torch.float32):
            layer.weight.data = state_tensors[w_key].to(dtype=dtype, device="cpu")
            continue

        raise RuntimeError(f"Don't know how to materialize weight for layer {lname} (missing/unknown format).")

# Keys used only for materialization; don't inject these into the model.
_MATERIALIZE_SIDECAR_SUFFIXES = (
    ".weight_scale", ".weight_scales", ".weight_scale_inv", ".weight_zero",
    ".qweight", ".qzeros", ".scales", ".g_idx", ".perm",
)

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

        materialize_fn(materialize_block, rel, dtype=dtype)

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
