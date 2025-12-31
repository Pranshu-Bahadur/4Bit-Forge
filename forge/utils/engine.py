from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import transformers
import re


def list_layers(block: nn.Module) -> Dict[str, nn.Module]:
    layers = {}
    print(block.named_modules())
    for n, m in block.named_modules():
        if isinstance(m, nn.Module) and re.search(r'.*(gate|up|down)_proj$'):
            if getattr(m, "weight", None):
                layers[n] = m
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

