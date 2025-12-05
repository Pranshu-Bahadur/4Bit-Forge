import torch
import triton
import triton.language as tl


@triton.jit
def build_group_meta_packed_triton(
    x_ptr,
    qmeta_ptr,
    group_size,
    n_bits,
    symmetric,
    x_stride0,
    x_stride1,
    qmeta_stride0,
    qmeta_stride1,
    BLOCK_SIZE: tl.constexpr
):
    # Moving pointers to respective group
    group_id = tl.program_id(0)
    x_ptr = x_ptr + group_id * x_stride0
    qmeta_ptr = qmeta_ptr + group_id * qmeta_stride0

    # Loading group
    group_offset = tl.arange(0, BLOCK_SIZE)
    group = tl.load(x_ptr + group_offset * x_stride1, mask=group_offset < group_size, other=0.0)

    # Computing maxq
    maxq = tl.cast(tl.exp2(tl.cast(n_bits, tl.float32)) - 1, tl.float32)
    
    # Computing group min and max
    group_min = tl.min(group)
    group_max = tl.max(group)

    # Computing scale
    eps = 1e-12
    if symmetric:
        amax = tl.max(tl.abs(group))
        s = (2.0 / maxq) * amax + eps
    else:
        s = (group_max - group_min) / maxq + eps
    
    # Computing log2_scale
    log2_scale = tl.log2(max(s, 1e-20))
    log2_scale = log2_scale * 256.0
    log2_scale = tl.minimum(tl.maximum(log2_scale, -32768.0), 32767.0)
    log2_scale_int = tl.cast(log2_scale, tl.int16)

    # Computing qzero
    if symmetric:
        q0 = 0.5 * (maxq + 1.0)
    else:
        q0 = -group_min / s
        q0 = tl.minimum(tl.maximum(q0, 0.0), maxq)
    qzero = tl.cast(q0, tl.uint8)
    
    # Computing flag
    flag = tl.cast(1 if symmetric else 0, tl.uint8)
    
    # Storing results (store as bytes)
    # Store log2_scale as int16 (2 bytes)
    tl.store(qmeta_ptr + 0 * qmeta_stride1, tl.cast(log2_scale_int & 0xFF, tl.uint8))
    tl.store(qmeta_ptr + 1 * qmeta_stride1, tl.cast((log2_scale_int >> 8) & 0xFF, tl.uint8))
    # Store qzero (1 byte)
    tl.store(qmeta_ptr + 2 * qmeta_stride1, qzero)
    # Store flag (1 byte)
    tl.store(qmeta_ptr + 3 * qmeta_stride1, flag)


    
def build_group_meta_packed(
    x_groups: torch.Tensor,
    bit_width: int,
    symmetric: bool 
):
    assert x_groups.ndim == 2, f"Expected x_groups to be 2 dimensional with (number_of_groups, group_size), but got dimension {x_groups.shape}"
    maxq = torch.tensor(2**bit_width - 1)
    qmeta = torch.empty(x_groups.shape[0], 4, device=x_groups.device, dtype=torch.uint8)
    group_size = x_groups.shape[1]
    
    grid = (x_groups.shape[0],)
    build_group_meta_packed_triton[grid](
        x_ptr=x_groups,
        qmeta_ptr=qmeta,
        group_size=group_size,
        n_bits=bit_width,
        symmetric=symmetric,
        x_stride0=x_groups.stride(0),
        x_stride1=x_groups.stride(1),
        qmeta_stride0=qmeta.stride(0),
        qmeta_stride1=qmeta.stride(1),
        BLOCK_SIZE=triton.next_power_of_2(group_size),
    )
    
    return qmeta, maxq

def mse_scale_groups_packed_triton():
    ...

def mse_scale_groups_packed(
    x_groups: torch.Tensor,
    p: torch.Tensor,
    qmeta_bytes: torch.Tensor,
):
    ...


