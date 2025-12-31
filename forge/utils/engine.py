from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn


def list_layers(block: nn.Module) -> Dict[str, nn.Linear]:
    layers = {}
    for n, m in block.named_modules():
        if isinstance(m, nn.Linear):
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


def forward(block, X, position_ids, N, bs, device, offload_device, act_update=False, rotary_embed=None):
    for s in range(0, N, bs):
        e = min(N, s + bs)

        # cached embeds
        x = torch.cat([X[i] for i in range(s, e)], dim=0).to(device, non_blocking=True)
        B = x.size(0)
        pos = position_ids.expand(B, -1)
        out = block(x, position_ids=pos) if rotary_embed is None else block(x, position_ids=pos, 
                                                                            position_embeddings=get_position_embeddings(rotary_embed, x, pos))
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
    

def ensure_rotary_ready(rotary_emb, config, *, device, max_seq_len: int, dtype: torch.dtype):
    """
    Fix RoPE modules that were created under init_empty_weights() and ended up with
    meta/None inv_freq or stale caches. Works for GPT-OSS + generally safe for DeepSeek.
    """
    if rotary_emb is None:
        return None

    # move module (best-effort; some RoPE modules have no params/buffers anyway)
    try:
        rotary_emb.to(device)
    except Exception:
        # if it ever complains about meta, we'll still overwrite inv_freq below
        pass

    if hasattr(rotary_emb, "inv_freq"):
        inv = getattr(rotary_emb, "inv_freq", None)
        bad = (inv is None) or (getattr(inv, "is_meta", False)) or (getattr(inv, "device", None) != torch.device(device))

        if bad:
            dim = getattr(rotary_emb, "dim", None)
            if dim is None:
                # common fallbacks
                dim = getattr(config, "rope_dim", None)
            if dim is None:
                dim = getattr(config, "head_dim", None)
            if dim is None:
                # last resort: hidden_size / num_attention_heads
                hs = getattr(config, "hidden_size", None)
                nh = getattr(config, "num_attention_heads", None)
                if hs is not None and nh is not None:
                    dim = hs // nh
            if dim is None:
                raise RuntimeError("Could not infer RoPE dim (rotary_emb.dim / config.head_dim / hidden_size//n_heads).")

            base = getattr(rotary_emb, "base", None)
            if base is None:
                base = getattr(config, "rope_theta", 10000.0)

            inv_freq = 1.0 / (float(base) ** (torch.arange(0, dim, 2, device=device).float() / float(dim)))

            # IMPORTANT: don't re-register if you can avoid it; just overwrite the buffer attr.
            rotary_emb.inv_freq = inv_freq

        # nuke caches so they regenerate correctly on first real forward
        if hasattr(rotary_emb, "cos_cached"):
            rotary_emb.cos_cached = None
        if hasattr(rotary_emb, "sin_cached"):
            rotary_emb.sin_cached = None

        # optional warmup (cheap-ish, but you can skip)
        try:
            L = min(int(max_seq_len), 2048)
            dummy_pos = torch.arange(0, L, device=device).unsqueeze(0)
            dummy_x = torch.zeros(1, L, int(getattr(rotary_emb, "dim", dummy_pos.numel())), device=device, dtype=dtype)
            _ = rotary_emb(dummy_x, dummy_pos)
        except Exception:
            pass

    return rotary_emb


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

