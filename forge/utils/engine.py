from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import transformers
import re

import torch
import torch.nn as nn
from typing import Dict, Any


class ParamSliceProxy:
    """
    Exposes a 2D .weight view into a 3D fused expert Parameter.
    layer.weight.copy_(...) will write back into the underlying Parameter storage.
    """
    def __init__(self, p3d: torch.Tensor, expert_idx: int):
        self._p3d = p3d
        self._e = expert_idx

    @property
    def weight(self) -> torch.Tensor:
        return self._p3d[self._e]  # 2D view


def list_layers(block: nn.Module) -> Dict[str, Any]:
    layers: Dict[str, Any] = {}

    # 1) Normal Linear modules (DeepSeek experts + non-expert projections)
    for n, m in block.named_modules():
        if n == "":
            continue
        if isinstance(m, nn.Linear):
            layers[n] = m

    # 2) GPT-OSS fused experts (Parameters on mlp.experts)
    # Look for 3D parameters named gate_up_proj / down_proj and expand per expert.
    fused = {}
    for n, p in block.named_parameters(recurse=True):
        # n looks like "mlp.experts.gate_up_proj" (NOT a module)
        if n.endswith("experts.gate_up_proj") and p.ndim == 3:
            fused["gate_up_proj"] = (n, p)
        elif n.endswith("experts.down_proj") and p.ndim == 3:
            fused["down_proj"] = (n, p)

    if "gate_up_proj" in fused and "down_proj" in fused:
        n_gu, p_gu = fused["gate_up_proj"]   # [E, H, 2D]
        n_dn, p_dn = fused["down_proj"]      # [E, D, H]
        E = p_gu.shape[0]
        if p_dn.shape[0] == E:
            # Emit names that match your routed-experts regex style:
            # mlp.experts.<e>.gate_up_proj / down_proj
            for e in range(E):
                layers[f"mlp.experts.{e}.gate_up_proj"] = ParamSliceProxy(p_gu, e)
                layers[f"mlp.experts.{e}.down_proj"]    = ParamSliceProxy(p_dn, e)

    return layers



def get_position_embeddings(rotary_emb: nn.Module, hidden_states: torch.Tensor, position_ids: torch.Tensor):
    """
    Generate RoPE embeddings for DeepSeek-V3 style blocks.
    Tries common HF signatures. Returns (cos, sin) or None.
    """
    try:
        return rotary_emb(hidden_states, position_ids)
    except TypeError:
        try:
            return rotary_emb(position_ids, seq_len=position_ids.shape[-1])
        except TypeError:
            return None

@torch.no_grad()
def dequantize_forge_full(dtype, qweight, scales, qzeros, group_size, device):

    qweight = qweight.to(device, non_blocking=True)

    if qweight.dim() != 2:
        raise ValueError(f"Expected qweight (R,C) unpacked, got shape={tuple(qweight.shape)}")

    R, C = qweight.shape
    G = (C + group_size - 1) // group_size

    # scales/qzeros are per-group for flattened [R*G]
    scales = scales.to(device, non_blocking=True).reshape(-1)
    qzeros = qzeros.to(device, non_blocking=True).reshape(-1)

    if scales.numel() != R * G or qzeros.numel() != R * G:
        raise ValueError(f"Expected scales/qzeros numel == R*G ({R*G}), "
                         f"got scales={scales.numel()}, qzeros={qzeros.numel()}")

    # reshape to [R, G]
    scales_rg = scales.view(R, G)
    qzeros_rg = qzeros.view(R, G)

    # expand each group to group_size columns -> [R, G*group_size] then crop -> [R, C]
    scale = scales_rg.repeat_interleave(group_size, dim=1)[:, :C]
    qzero = qzeros_rg.repeat_interleave(group_size, dim=1)[:, :C]

    w = (qweight.to(torch.float32) - qzero.to(torch.float32)) * scale.to(torch.float32)
    return w.to(dtype), scale, qzero


def forward(block, X, position_ids, N, bs, device, offload_device, act_update=False, rotary_emb=None):
    for s in range(0, N, bs):
        e = min(N, s + bs)

        # cached embeds
        x = torch.cat([X[i] for i in range(s, e)], dim=0).to(device, non_blocking=True)
        B = x.size(0)
        pos = position_ids.expand(B, -1)
        if rotary_emb is None:
            out = block(x, position_ids=pos)
        else:
            pos_emb = rotary_emb(x, pos)  # returns (cos, sin) for GPT-OSS :contentReference[oaicite:2]{index=2}
            try:
                out = block(x, position_ids=pos, position_embeddings=pos_emb)
            except TypeError:
                # DeepSeek-like blocks often compute RoPE internally and don't accept position_embeddings
                out = block(x, position_ids=pos)
        out = out[0] if isinstance(out, (tuple, list)) else out
        out = out.to(offload_device) if offload_device is not None else out
        if act_update:
            for j in range(B):
                X[s + j] = out[j:j+1].contiguous()

        del x, pos, out
    if act_update:
        return X
    else:
        return None
    
def ensure_rotary_emb(model, config, device):
    # find rotary_emb in common places
    rotary = None
    try:
        rotary = model.model.rotary_emb
    except Exception:
        try:
            rotary = model.rotary_emb
        except Exception:
            rotary = None

    if rotary is None:
        return None

    # If init_empty_weights put buffers on meta, regenerate them on real device.
    inv = getattr(rotary, "inv_freq", None)
    if inv is None or getattr(inv, "is_meta", False):
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        # Match HF logic: use rotary.rope_type if present else config.rope_scaling rope_type else default
        rope_type = getattr(rotary, "rope_type", None)
        if rope_type is None:
            rs = getattr(config, "rope_scaling", None) or {}
            rope_type = rs.get("rope_type", rs.get("type", "default"))

        rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
        inv_freq, attention_scaling = rope_init_fn(config, device)

        rotary.register_buffer("inv_freq", inv_freq, persistent=False)
        rotary.original_inv_freq = rotary.inv_freq
        rotary.attention_scaling = attention_scaling

        # nuke any caches if the class uses them
        for k in ("cos_cached", "sin_cached", "_cos_cached", "_sin_cached"):
            if hasattr(rotary, k):
                setattr(rotary, k, None)

    rotary.to(device)
    return rotary



def find_rotary_emb(model):
    # common roots
    roots = [
        model,
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "model", None),
        getattr(model, "transformer", None),
    ]

    for r in roots:
        if r is None:
            continue
        rot = getattr(r, "rotary_emb", None)
        if rot is not None:
            return rot

    # fallback: look inside first layer attention modules
    for r in roots:
        if r is None:
            continue
        layers = getattr(r, "layers", None) or getattr(r, "h", None)
        if layers is None:
            continue
        for i in range(min(len(layers), 2)):
            blk = layers[i]
            sa = getattr(blk, "self_attn", None) or getattr(blk, "attn", None)
            if sa is None:
                continue
            rot = getattr(sa, "rotary_emb", None)
            if rot is not None:
                return rot

    return None

