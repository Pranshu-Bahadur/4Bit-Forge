#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pack_quantized.py

Takes:
  1) A base HF checkpoint (safetensors + model.safetensors.index.json)
  2) A 4Bit-Forge quantized weights directory produced by your quant.py (--save_dir),
     which contains:
        - metadata.pt
        - per-layer subdirs each containing quantized_weight.pt

Produces:
  - A "packed" model directory that stores most weights as float (bf16/fp16)
    but replaces selected Linear weights with compressed-tensors packed int32 format.
  - A new model.safetensors.index.json mapping.

This script is intentionally "quant.py-aligned":
  - Uses forge.utils.io.load_safetensors_index + ShardDiskWindow + ShardLRU
  - Does not assume a fixed 000163 shard count or filename template.
"""

import os
import gc
import re
import json
import shutil
import argparse
from collections import defaultdict
from typing import Optional, Any, Dict, Set, Tuple
from tqdm import tqdm

import torch
from safetensors.torch import save_file, safe_open

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from compressed_tensors.compressors import pack_to_int32

import forge.utils


# ---------------------------
# args
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_name_or_path", type=str, required=True,
                   help="Base model HF repo id or local dir containing safetensors + index json.")
    p.add_argument("--quantized_model_path", type=str, required=True,
                   help="Directory produced by your 4BF quant.py (--save_dir).")
    p.add_argument("--packed_model_path", type=str, required=True,
                   help="Output directory for packed model.")

    p.add_argument("--dtype", default="float16", type=str, choices=["float16", "bfloat16"],
                   help="Float dtype to store non-quantized weights.")
    p.add_argument("--assume_sym", action="store_true",
                   help="Force symmetric packing (omit zero-point). Default: True if metadata indicates sym.")
    p.add_argument("--copy_modeling_file", action="store_true",
                   help="If set, tries to copy modeling_deepseek.py from base dir into output.")

    # IO tuning (same spirit as your quant.py)
    p.add_argument("--hf_tmp_dir", type=str, default="/tmp/hf_jit",
                   help="Tmp dir for shard window downloads.")
    p.add_argument("--lru_ram_gb", type=float, default=0.0,
                   help="RAM-side shard tensor cache size (GB). 0 disables caching.")

    p.add_argument("--disk_max_shards", type=int, default=8,
                   help="Max shard files to keep on disk (if shard_bytes=0).")
    p.add_argument("--disk_safety_gb", type=float, default=2.0,
                   help="Safety free disk to keep (GB).")
    p.add_argument("--disk_shard_gb", type=float, default=0.0,
                   help="If >0, enables reserve-aware cap based on estimated shard size.")
    p.add_argument("--reserve_gb", type=float, default=0.0,
                   help="Reserve bytes for upcoming output writes (GB).")

    return p.parse_args()


# ---------------------------
# helpers
# ---------------------------

def _parse_block_idx_from_layer_name(layer_dir_name: str) -> int:
    """
    Expects layer_dir_name like:
      'model.layers.12.mlp.experts.0.up_proj'  (best)
    We also accept:
      'layers.12.mlp.experts.0.up_proj'
    """
    m = re.search(r"(?:^|\.)(?:model\.)?layers\.(\d+)\.", layer_dir_name)
    if not m:
        raise ValueError(
            f"Could not parse block index from quantized layer dir: '{layer_dir_name}'. "
            "Expected it to contain 'model.layers.<idx>.' or 'layers.<idx>.'"
        )
    return int(m.group(1))


def _collect_prefix_tensors(
    repo_id: str,
    weight_map: Dict[str, str],
    prefix: str,
    tmp_dir: str,
    *,
    lru,
    disk_window,
    reserve_bytes: int,
) -> Dict[str, torch.Tensor]:
    """
    Collect all tensors in checkpoint that start with `prefix`.
    Uses ShardLRU + ShardDiskWindow (same as jit_load_prefix_to_cpu) but returns a dict.

    NOTE: keys in returned dict are FULL checkpoint names (e.g. "model.layers.12....").
    """
    needed = [name for name in weight_map.keys() if name.startswith(prefix)]
    if not needed:
        return {}

    by_shard: Dict[str, list[str]] = defaultdict(list)
    for name in needed:
        by_shard[weight_map[name]].append(name)

    pinned_filenames: Set[str] = set(by_shard.keys())
    out: Dict[str, torch.Tensor] = {}

    for shard_name, names in by_shard.items():
        tensors = lru.get(shard_name) or {}
        missing = [n for n in names if n not in tensors]

        if missing:
            shard_path = disk_window.download_shard(
                repo_id,
                shard_name,
                reserve_bytes=int(reserve_bytes),
                pinned_filenames=pinned_filenames,
            )
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                keys = set(f.keys())
                for tname in missing:
                    if tname not in keys:
                        raise KeyError(f"Tensor not found in shard: {tname} (shard={shard_name})")
                    tensors[tname] = f.get_tensor(tname)

            total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
            lru.put(shard_name, tensors, total_bytes)

        for tname in names:
            out[tname] = tensors[tname]

    return out


def _collect_exact_tensors(
    repo_id: str,
    weight_map: Dict[str, str],
    keys: list[str],
    tmp_dir: str,
    *,
    lru,
    disk_window,
    reserve_bytes: int,
) -> Dict[str, torch.Tensor]:
    """
    Collect an explicit list of tensor names from checkpoint.
    """
    keys = [k for k in keys if k in weight_map]
    if not keys:
        return {}

    by_shard: Dict[str, list[str]] = defaultdict(list)
    for name in keys:
        by_shard[weight_map[name]].append(name)

    pinned_filenames: Set[str] = set(by_shard.keys())
    out: Dict[str, torch.Tensor] = {}

    for shard_name, names in by_shard.items():
        tensors = lru.get(shard_name) or {}
        missing = [n for n in names if n not in tensors]

        if missing:
            shard_path = disk_window.download_shard(
                repo_id,
                shard_name,
                reserve_bytes=int(reserve_bytes),
                pinned_filenames=pinned_filenames,
            )
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                keys_set = set(f.keys())
                for tname in missing:
                    if tname not in keys_set:
                        raise KeyError(f"Tensor not found in shard: {tname} (shard={shard_name})")
                    tensors[tname] = f.get_tensor(tname)

            total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
            lru.put(shard_name, tensors, total_bytes)

        for tname in names:
            out[tname] = tensors[tname]

    return out


def _maybe_dequantize_int_weight_inplace(state: Dict[str, torch.Tensor], base: str, dtype: torch.dtype) -> None:
    """
    Matches the conventions you outlined:

    - base+".weight" + base+".weight_scales"   (+ optional weight_zero)
    - base+".weight" + base+".weight_scale"    (+ optional weight_zero)
    - base+".weight" + base+".weight_scale_inv" (+ optional weight_zero)

    Dequantizes ONLY when weight is integer type.
    Leaves float weights unchanged.
    """
    w_key = base + ".weight"
    if w_key not in state:
        return

    w = state[w_key]
    #if w.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32):
        # also handles fp8/bf16/fp16 etc: just cast fp8 to dtype elsewhere if you want
    #    return

    zero_key = base + ".weight_zero"
    z = state.get(zero_key, None)
    if z is not None:
        zf = z.to(torch.float32)
    else:
        zf = None

    # prefer weight_scales then weight_scale then weight_scale_inv
    if (base + ".weight_scales") in state:
        s = state[base + ".weight_scales"].to(torch.float32)
        wf = w.to(torch.float32)
        if zf is not None:
            wf = (wf - zf) * s
        else:
            wf = wf * s
        state[w_key] = wf.to(dtype=dtype)
        return

    if (base + ".weight_scale") in state:
        s = state[base + ".weight_scale"].to(torch.float32)
        wf = w.to(torch.float32)
        if zf is not None:
            wf = (wf - zf) * s
        else:
            wf = wf * s
        state[w_key] = wf.to(dtype=dtype)
        return

    if (base + ".weight_scale_inv") in state:
        s_inv = state[base + ".weight_scale_inv"].to(torch.float32)
        wf = w.to(torch.float32)
        # your convention: multiply by (1/s_inv)
        inv = 1.0 / (s_inv + 1e-30)
        if zf is not None:
            wf = (wf - zf) * inv
        else:
            wf = wf * inv
        state[w_key] = wf.to(dtype=dtype)
        return


def _cast_fp8_to_dtype_inplace(state: Dict[str, torch.Tensor], dtype: torch.dtype) -> None:
    """
    Best-effort: cast fp8 tensors to dtype (bf16/fp16) since many inference stacks
    won't like fp8 weights stored in safetensors.
    """
    for k, v in list(state.items()):
        if str(v.dtype).startswith("torch.float8"):
            state[k] = v.to(dtype=dtype)


def pack_weight(
    weight: Dict[str, torch.Tensor],
    bits: int,
    sym: bool,
    group_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Packs your 4BF qweight/scale/zero into compressed-tensors format.

    Assumes:
      - weight["qweight"] is int8/uint8 holding unpacked int4 values in [0, 15]
      - weight["zero"] is int8/uint8 of per-group zero points (broadcastable)
      - weight["scale"] is float (per-group)
    """
    qweight = weight["qweight"]
    scale = weight["scale"]
    zero = weight["zero"]

    group_size = int(group_size or qweight.shape[-1])

    # shift into signed centered values before packing
    # note: zero stored per-group; repeat across group_size along last dim
    zero_rep = zero.repeat_interleave(group_size, dim=-1).to(torch.int8)
    q_shift = qweight.to(torch.int8) - zero_rep

    q_packed = pack_to_int32(q_shift, bits)

    out = {
        "weight_packed": q_packed,
        "weight_shape": torch.tensor(qweight.shape, dtype=torch.int64),
        "weight_scale": scale,
    }
    if not sym:
        out["weight_zero_point"] = zero
    return out


def prepare_quantization_config(bits: int, group_size: int, quantize_only_experts: bool) -> Dict[str, Any]:
    ignored_modules = ["lm_head"]
    if quantize_only_experts:
        ignored_modules += ["re:.*self_attn.*", "re:.*shared_experts.*", "re:.*mlp\\.(gate|up|gate_up|down)_proj.*"]
    return {
        "config_groups": {
            "group_0": {
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": int(group_size),
                    "num_bits": int(bits),
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "group",
                    "symmetric": True,
                    "type": "int",
                },
            }
        },
        "format": "pack-quantized",
        "ignore": ignored_modules,
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }


# ---------------------------
# main
# ---------------------------

def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    os.makedirs(args.packed_model_path, exist_ok=True)

    # Set HF caches to tmp dir (same as quant.py spirit)
    os.environ.setdefault("HF_HOME", args.hf_tmp_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", args.hf_tmp_dir)
    os.environ.setdefault("HF_HUB_CACHE", args.hf_tmp_dir)

    # Load base config + meta model
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # meta dtype doesn't matter much
        ).eval()
        model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Load 4BF quant metadata
    meta_path = os.path.join(args.quantized_model_path, "metadata.pt")
    meta = torch.load(meta_path, map_location="cpu")
    bits = int(meta["bits"])
    group_size = int(meta["group_size"])
    quantize_only_experts = bool(meta.get("quantize_only_experts", meta.get("quantize_only_routed_experts", False)))

    # Sym: your pipeline mostly uses sym; allow override
    sym = True
    if args.assume_sym:
        sym = True

    # Build mapping: block_idx -> [layer_dir_names]
    quantized_layer_names: Dict[int, list[str]] = defaultdict(list)
    for entry in sorted(os.listdir(args.quantized_model_path)):
        full = os.path.join(args.quantized_model_path, entry)
        if os.path.isdir(full):
            bidx = _parse_block_idx_from_layer_name(entry)
            quantized_layer_names[bidx].append(entry)

    # Base checkpoint index
    shard_ids = forge.utils.io.load_safetensors_index(args.model_name_or_path, tmp_dir=args.hf_tmp_dir)
    weight_map = shard_ids["weight_map"]

    # IO objects
    lru = forge.utils.io.ShardLRU(max_bytes=int(args.lru_ram_gb * (1024 ** 3)))

    disk_cfg = forge.utils.io.DiskWindowConfig(
        shard_bytes=int(args.disk_shard_gb * (1024 ** 3)) if args.disk_shard_gb > 0 else 0,
        safety_bytes=int(args.disk_safety_gb * (1024 ** 3)),
        max_shards=int(args.disk_max_shards),
        use_lru_touch=True,
    )
    disk_window = forge.utils.io.ShardDiskWindow(args.hf_tmp_dir, disk_cfg)
    reserve_bytes = int(args.reserve_gb * (1024 ** 3))

    # Output sharding: embed + each block + final
    num_output_shards = len(model.model.layers) + 2
    out_weight_map: Dict[str, str] = {}

    def out_name(shard_id: int) -> str:
        return f"model-{shard_id:05}-of-{num_output_shards:05}.safetensors"

    # ---- shard 1: embeddings ----
    emb_tensors = _collect_exact_tensors(
        args.model_name_or_path,
        weight_map,
        ["model.embed_tokens.weight"],
        args.hf_tmp_dir,
        lru=lru,
        disk_window=disk_window,
        reserve_bytes=reserve_bytes,
    )
    if "model.embed_tokens.weight" not in emb_tensors:
        raise KeyError("Base checkpoint is missing model.embed_tokens.weight")

    shard1 = out_name(1)
    save_file({"model.embed_tokens.weight": emb_tensors["model.embed_tokens.weight"]},
              os.path.join(args.packed_model_path, shard1))
    out_weight_map["model.embed_tokens.weight"] = shard1

    del emb_tensors
    gc.collect()

    # ---- blocks ----
    for block_idx, _block in tqdm(enumerate(model.model.layers),
                                  desc="Packing transformer blocks",
                                  total=len(model.model.layers)):
        shard_id = block_idx + 2  # shard 2..(L+1)
        prefix = f"model.layers.{block_idx}."

        block_state = _collect_prefix_tensors(
            args.model_name_or_path,
            weight_map,
            prefix,
            args.hf_tmp_dir,
            lru=lru,
            disk_window=disk_window,
            reserve_bytes=reserve_bytes,
        )

        if not block_state:
            raise KeyError(f"No tensors found for prefix: {prefix}")

        # best-effort dtype normalization
        #_cast_fp8_to_dtype_inplace(block_state, dtype=dtype)

        # integer->float dequantization for common patterns (optional)
        # We only attempt it for tensors that look like "...<something>.weight"
        # based on keys present in the dict.
        # Example base: "model.layers.12.mlp.experts.0.up_proj"
        bases = set()
        for k in block_state.keys():
            if k.endswith(".weight"):
                bases.add(k[:-len(".weight")])
        for base in bases:
            _maybe_dequantize_int_weight_inplace(block_state, base, dtype=dtype)

        # Replace quantized layers in this block with packed int32 tensors
        for layer_name in quantized_layer_names.get(block_idx, []):
            w_pt = os.path.join(args.quantized_model_path, layer_name, "quantized_weight.pt")
            q = torch.load(w_pt, map_location="cpu", weights_only=True)

            packed = pack_weight(q, bits=bits, sym=sym, group_size=group_size)

            # remove original float weight + common scale sidecars if present
            w_key = f"{layer_name}.weight"
            block_state.pop(w_key, None)
            block_state.pop(f"{layer_name}.weight_scale_inv", None)
            block_state.pop(f"{layer_name}.weight_scale", None)
            block_state.pop(f"{layer_name}.weight_scales", None)
            block_state.pop(f"{layer_name}.weight_zero", None)

            for k, v in packed.items():
                block_state[f"{layer_name}.{k}"] = v

        # Save shard
        shard_path = os.path.join(args.packed_model_path, out_name(shard_id))
        save_file(block_state, shard_path)

        for k in block_state.keys():
            out_weight_map[k] = os.path.basename(shard_path)

        del block_state
        gc.collect()

    # ---- final shard: lm_head + norm ----
    final_id = num_output_shards
    final_keys = ["lm_head.weight", "model.norm.weight"]
    final_tensors = _collect_exact_tensors(
        args.model_name_or_path,
        weight_map,
        final_keys,
        args.hf_tmp_dir,
        lru=lru,
        disk_window=disk_window,
        reserve_bytes=reserve_bytes,
    )
    missing = [k for k in final_keys if k not in final_tensors]
    if missing:
        raise KeyError(f"Base checkpoint missing required final tensors: {missing}")

    final_name = out_name(final_id)
    save_file(final_tensors, os.path.join(args.packed_model_path, final_name))
    for k in final_tensors.keys():
        out_weight_map[k] = final_name

    # Write safetensors index
    with open(os.path.join(args.packed_model_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": out_weight_map}, f)

    # Add quantization_config to config + save
    config.quantization_config = prepare_quantization_config(bits, group_size, quantize_only_experts)
    config.save_pretrained(args.packed_model_path)
    model.generation_config.save_pretrained(args.packed_model_path)
    tokenizer.save_pretrained(args.packed_model_path)

    # Optional: copy modeling file (only works when base is local dir)
    if args.copy_modeling_file and os.path.isdir(args.model_name_or_path):
        src = os.path.join(args.model_name_or_path, "modeling_deepseek.py")
        if os.path.exists(src):
            shutil.copy(src, args.packed_model_path)

    print(f"[OK] Packed model written to: {args.packed_model_path}")


if __name__ == "__main__":
    main()