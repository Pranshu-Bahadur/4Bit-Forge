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
