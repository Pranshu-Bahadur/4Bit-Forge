%%writefile /content/export_packed.py
#!/usr/bin/env python3
"""
Pack SparseGPTQ 1:4 artifacts into your custom Wpair (ulonglong2) format using:
  forge.backends.cuda.kernels.pack_sparsegptq14_to_u64x2_cuda

Input assumptions (per your notes):
- qweight is [R, C]  (pre-nibble int8/uint8; values either 0..15 or -8..7)
- M is [G32, R] uint32
- scales is [G32, R] float/half/bf16 (we cast to float32)
- (Optional) qzeros may exist, ignored for symmetric path

Outputs (per block):
- gate_up packed (stacked gate+up):  W13_all: [E, G2_7168, 4096, 2] uint64
- down packed:                      W2_all:  [E, G2_2048, 7168, 2] uint64

Saved as safetensors:
  out_dir/block.<id>/experts_gate_up_Wpair_u64.safetensors  (key: "Wpair_u64")
  out_dir/block.<id>/experts_down_Wpair_u64.safetensors     (key: "Wpair_u64")
  out_dir/block.<id>/meta.json

Run:
  python /content/export_packed.py \
    --quant_dir /path/to/Quantized \
    --out_dir /path/to/Packed \
    --device cuda:0 \
    --num_experts 128
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import torch
from safetensors.torch import save_file


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _import_pack_fn():
    # Try a couple common spellings; fail loudly if not found.
    try:
        from forge.backends.cuda.kernels import pack_sparsegptq14_to_u64x2_cuda as fn
        return fn
    except Exception as e1:
        try:
            from forge.backends.cuda.kernels import pack_sparsegptq14_to_u64x2 as fn
            return fn
        except Exception as e2:
            raise ImportError(
                "Could not import pack kernel. Expected one of:\n"
                "  forge.backends.cuda.kernels.pack_sparsegptq14_to_u64x2_cuda\n"
                "  forge.backends.cuda.kernels.pack_sparsegptq14_to_u64x2\n"
                f"Errors:\n  {e1}\n  {e2}"
            )


def _load_sparse14_pt(pt_path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (qweight_rc, scales_g32r, M_g32r) all on CPU.
    Accepts a few key variants.
    """
    d: Dict[str, Any] = torch.load(pt_path, map_location="cpu")

    # qweight
    if "qweight" in d:
        qw = d["qweight"]
    elif "qw" in d:
        qw = d["qw"]
    else:
        raise KeyError(f"{pt_path} missing qweight. keys={list(d.keys())}")

    # scales
    if "scale" in d:
        sc = d["scale"]
    elif "scales" in d:
        sc = d["scales"]
    else:
        raise KeyError(f"{pt_path} missing scale/scales. keys={list(d.keys())}")

    # mask M
    if "M" in d:
        M = d["M"]
    elif "mask" in d:
        M = d["mask"]
    else:
        raise KeyError(f"{pt_path} missing M/mask. keys={list(d.keys())}")

    if qw.dim() != 2:
        raise ValueError(f"{pt_path}: qweight must be [R,C], got {tuple(qw.shape)}")
    if M.dim() != 2:
        raise ValueError(f"{pt_path}: M must be [G32,R], got {tuple(M.shape)}")
    if sc.dim() != 2:
        raise ValueError(f"{pt_path}: scales must be [G32,R], got {tuple(sc.shape)}")

    # Normalize dtypes for later
    qw = qw.contiguous()
    sc = sc.contiguous().to(torch.float32)
    M = M.contiguous().to(torch.uint32)

    return qw, sc, M


def _to_q_u8_offset_binary(qw_rc: torch.Tensor) -> torch.Tensor:
    """
    Convert pre-nibble qweight into uint8 nibbles 0..15 such that value = nibble - 8.
    Supports:
      - int8 in [-8..7]
      - uint8/int8 already in [0..15]
    """
    if qw_rc.dtype == torch.uint8:
        return (qw_rc & 0x0F).contiguous()

    qw_i32 = qw_rc.to(torch.int32)
    mn = int(qw_i32.min().item())
    mx = int(qw_i32.max().item())

    if mn < 0:
        # assume signed int4 domain [-8..7]
        out = (qw_i32 + 8).clamp(0, 15).to(torch.uint8)
    else:
        # assume already 0..15 (or wider, but we mask)
        out = (qw_i32 & 0x0F).to(torch.uint8)

    return out.contiguous()


def _pad_C_to_64(qw_rc_u8: torch.Tensor, C_pad: int) -> torch.Tensor:
    """
    qw_rc_u8 is [R,C]. Pad columns to C_pad with zeros.
    """
    R, C = qw_rc_u8.shape
    if C == C_pad:
        return qw_rc_u8
    out = torch.zeros((R, C_pad), dtype=torch.uint8)
    out[:, :C] = qw_rc_u8
    return out.contiguous()


def _pad_G32(sc_g32r: torch.Tensor, M_g32r: torch.Tensor, G32_pad: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    sc_g32r, M_g32r are [G32,R]. Pad first dim to G32_pad with zeros.
    """
    G32, R = sc_g32r.shape
    if G32 == G32_pad:
        return sc_g32r, M_g32r
    sc2 = torch.zeros((G32_pad, R), dtype=torch.float32)
    M2 = torch.zeros((G32_pad, R), dtype=torch.uint32)
    sc2[:G32, :] = sc_g32r
    M2[:G32, :] = M_g32r
    return sc2.contiguous(), M2.contiguous()


def _pack_one(
    pack_fn,
    qw_rc: torch.Tensor,
    sc_g32r: torch.Tensor,
    M_g32r: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """
    Calls your CUDA pack kernel and returns Wpair_u64 on CPU with shape [G2, R, 2] uint64.
    """
    qw_u8 = _to_q_u8_offset_binary(qw_rc)

    R, C = qw_u8.shape
    C_pad = ceil_div(C, 64) * 64
    G32_pad = ceil_div(C_pad, 32)

    qw_u8 = _pad_C_to_64(qw_u8, C_pad=C_pad)

    # validate/pad M/scales to match padded C
    if sc_g32r.shape[1] != R or M_g32r.shape[1] != R:
        raise ValueError(f"sc/M R mismatch: sc={tuple(sc_g32r.shape)} M={tuple(M_g32r.shape)} R={R}")
    sc_g32r, M_g32r = _pad_G32(sc_g32r, M_g32r, G32_pad=G32_pad)

    # Move to GPU and pack
    qw_u8 = qw_u8.to(device=device, non_blocking=True)
    sc_g32r = sc_g32r.to(device=device, non_blocking=True)
    M_g32r = M_g32r.to(device=device, non_blocking=True)

    # pack_fn expects:
    #   qweight_rc: [R,C] uint8
    #   M: [G32,R] uint32
    #   scales: [G32,R] float32
    Wpair_u64 = pack_fn(qw_u8, M_g32r, sc_g32r)  # -> uint64 [G2,R,2] (CUDA)

    torch.cuda.synchronize(device)
    return Wpair_u64.to("cpu").contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant_dir", type=str, required=True, help="Root Quantized dir containing block.N/..")
    ap.add_argument("--out_dir", type=str, required=True, help="Output dir for packed tensors")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--num_experts", type=int, default=128)
    ap.add_argument("--gate_name", type=str, default="gate_proj")
    ap.add_argument("--up_name", type=str, default="up_proj")
    ap.add_argument("--down_name", type=str, default="down_proj")
    ap.add_argument("--in_gate_up", type=int, default=7168)   # C for gate/up
    ap.add_argument("--out_gate_up", type=int, default=2048)  # R for gate/up (before stacking)
    ap.add_argument("--in_down", type=int, default=2048)      # C for down
    ap.add_argument("--out_down", type=int, default=7168)     # R for down
    ap.add_argument("--pt_name", type=str, default="sparse14quantized_weight.pt")
    args = ap.parse_args()

    quant_dir = Path(args.quant_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This packer uses a CUDA kernel; --device must be cuda:*")

    pack_fn = _import_pack_fn()

    # Expect block.<id> folders
    block_dirs = sorted([p for p in quant_dir.iterdir() if p.is_dir() and re.fullmatch(r"block\.\d+", p.name)],
                        key=lambda p: int(p.name.split(".")[1]))
    if not block_dirs:
        raise RuntimeError(f"No block.N directories found in {quant_dir}")

    E = int(args.num_experts)

    # Known shapes => known packed G2
    G2_w13 = ceil_div(args.in_gate_up, 64)  # for C=7168 => 112
    R_w13 = 2 * args.out_gate_up            # stacked => 4096
    G2_w2 = ceil_div(args.in_down, 64)      # for C=2048 => 32
    R_w2 = args.out_down                    # 7168

    for block_path in block_dirs:
        block_id = int(block_path.name.split(".")[1])
        print(f"[block {block_id}] scanning {block_path}")

        # Preallocate CPU outputs (big but manageable and avoids 15k files)
        W13_all = torch.empty((E, G2_w13, R_w13, 2), dtype=torch.uint64, device="cpu")
        W2_all = torch.empty((E, G2_w2, R_w2, 2), dtype=torch.uint64, device="cpu")

        # Fill with zeros by default (in case some experts missing on disk)
        W13_all.zero_()
        W2_all.zero_()

        # We expect dirs like: mlp.experts.<eid>.gate_proj  (etc) under block dir
        for eid in range(E):
            base = f"mlp.experts.{eid}."

            gate_dir = block_path / f"{base}{args.gate_name}"
            up_dir   = block_path / f"{base}{args.up_name}"
            down_dir = block_path / f"{base}{args.down_name}"

            gate_pt = gate_dir / args.pt_name
            up_pt   = up_dir / args.pt_name
            down_pt = down_dir / args.pt_name

            if not gate_pt.exists() or not up_pt.exists() or not down_pt.exists():
                # leave zeros if missing
                continue

            # Load CPU artifacts
            qw_g, sc_g, M_g = _load_sparse14_pt(gate_pt)
            qw_u, sc_u, M_u = _load_sparse14_pt(up_pt)
            qw_d, sc_d, M_d = _load_sparse14_pt(down_pt)

            # Validate shapes for stacking
            if qw_g.shape != qw_u.shape:
                raise ValueError(f"eid={eid}: gate/up qweight shape mismatch: {qw_g.shape} vs {qw_u.shape}")
            if sc_g.shape != sc_u.shape:
                raise ValueError(f"eid={eid}: gate/up scales shape mismatch: {sc_g.shape} vs {sc_u.shape}")
            if M_g.shape != M_u.shape:
                raise ValueError(f"eid={eid}: gate/up M shape mismatch: {M_g.shape} vs {M_u.shape}")

            # Stack R: qweight is [R,C] so cat on dim=0
            qw_w13 = torch.cat([qw_g, qw_u], dim=0).contiguous()        # [4096, 7168]
            # M/scales are [G32,R] so cat on dim=1
            sc_w13 = torch.cat([sc_g, sc_u], dim=1).contiguous()        # [G32, 4096]
            M_w13  = torch.cat([M_g,  M_u],  dim=1).contiguous()        # [G32, 4096]

            # Pack on GPU (streamed per-expert)
            with torch.inference_mode():
                W13 = _pack_one(pack_fn, qw_w13, sc_w13, M_w13, device=device)  # CPU [G2,4096,2]
                W2  = _pack_one(pack_fn, qw_d,   sc_d,   M_d,   device=device)  # CPU [G2,7168,2]

            # Store into preallocated big tensors
            if W13.shape[0] != G2_w13 or W13.shape[1] != R_w13 or W13.shape[2] != 2:
                raise ValueError(f"eid={eid}: W13 packed shape unexpected: {tuple(W13.shape)}")
            if W2.shape[0] != G2_w2 or W2.shape[1] != R_w2 or W2.shape[2] != 2:
                raise ValueError(f"eid={eid}: W2 packed shape unexpected: {tuple(W2.shape)}")

            W13_all[eid].copy_(W13)
            W2_all[eid].copy_(W2)

            if (eid % 8) == 0:
                print(f"  packed expert {eid}/{E-1}")

        # Write outputs for this block
        out_block = out_dir / f"block.{block_id}"
        out_block.mkdir(parents=True, exist_ok=True)

        save_file({"Wpair_u64": W13_all}, str(out_block / "experts_gate_up_Wpair_u64.safetensors"))
        save_file({"Wpair_u64": W2_all},  str(out_block / "experts_down_Wpair_u64.safetensors"))

        meta = {
            "block_id": block_id,
            "num_experts": E,
            "gate_up": {"C": args.in_gate_up, "R_gate": args.out_gate_up, "R_stacked": R_w13, "G2": G2_w13},
            "down":    {"C": args.in_down,    "R": args.out_down,        "G2": G2_w2},
            "format": "scale16|idx16|qw32 packed into u64, two u64 per g2 => ulonglong2",
        }
        (out_block / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[block {block_id}] wrote -> {out_block}")

    print(f"[OK] Packed all blocks -> {out_dir}")


if __name__ == "__main__":
    main()
