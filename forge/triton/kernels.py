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
    assert x_groups.is_cuda, f"x_groups must be CUDA"
    assert x_groups.ndim == 2, f"Expected x_groups to be 2 dimensional with (number_of_groups, group_size), but got dimension {x_groups.shape}"
    maxq = torch.tensor(2**bit_width - 1)
    qmeta = torch.empty(x_groups.shape[0], 4, device=x_groups.device, dtype=torch.uint8)
    group_size = x_groups.shape[1]
    block_size = triton.next_power_of_2(group_size)
    assert block_size < 1024, "Cannot use triton kernel with group size larger than 1024"
    
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
        BLOCK_SIZE=block_size,
    )
    
    return qmeta, maxq

@triton.jit
def mse_scale_groups_packed_triton(
    x_ptr,
    qmeta_ptr,
    p_ptr,
    group_size: tl.constexpr,
    P: tl.constexpr,
    maxq: tl.constexpr,
    norm,
    is_l2_norm: tl.constexpr,
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
    
    # Load existing qmeta to get base scale and qzero
    log2_scale_byte0 = tl.load(qmeta_ptr + 0 * qmeta_stride1)
    log2_scale_byte1 = tl.load(qmeta_ptr + 1 * qmeta_stride1)
    qzero = tl.load(qmeta_ptr + 2 * qmeta_stride1)
    
    # Decode log2_scale from int16 stored as 2 bytes
    log2_scale_int = tl.cast(log2_scale_byte0, tl.int32) | (tl.cast(log2_scale_byte1, tl.int32) << 8)
    # Handle sign extension for int16
    log2_scale_int = tl.where(log2_scale_int >= 32768, log2_scale_int - 65536, log2_scale_int)
    
    # Decode scale from Q8.8 fixed-point
    log2_scale_fp = tl.cast(log2_scale_int, tl.float32) / 256.0
    base_s = tl.exp2(log2_scale_fp)
    q0 = tl.cast(qzero, tl.float32)
    
    # Load group data
    group_offset = tl.arange(0, BLOCK_SIZE)
    mask = group_offset < group_size
    group = tl.load(x_ptr + group_offset * x_stride1, mask=mask, other=0.0)
    
    # Initialize best loss and scale
    best_loss = tl.cast(1e30, tl.float32)
    best_s = base_s
    
    # Grid search over scale candidates
    for k in range(P):
        # Load scale multiplier from p array
        p_val = tl.load(p_ptr + k)
        s = base_s * p_val
        rcp = 1.0 / s
        
        # Quantize and compute error for all elements
        q = group * rcp + q0
        q = tl.maximum(tl.minimum(q, maxq), 0.0)
        q = tl.cast(q, tl.int32)
        
        # Dequantize
        deq = (q - q0) * s
        
        # Compute error
        err = tl.abs(deq - group)
        
        if is_l2_norm:
            err = err * err
        else:
            err = tl.exp(tl.log(err) * norm)
        
        # Sum error over group (only valid elements)
        err = tl.where(mask, err, 0.0)
        loss = tl.sum(err)
        
        # Update best if this is better
        if loss < best_loss:
            best_loss = loss
            best_s = s
    
    # Thread 0 writes back the best scale
    if tl.program_id(1) == 0:
        # Encode scale back to Q8.8 fixed-point
        log2_scale = tl.log2(tl.maximum(best_s, 1e-20))
        log2_scale_fp = log2_scale * 256.0
        log2_scale_fp = tl.minimum(tl.maximum(log2_scale_fp, -32768.0), 32767.0)
        log2_scale_int = tl.cast(log2_scale_fp, tl.int32)
        
        # Store as 2 bytes
        tl.store(qmeta_ptr + 0 * qmeta_stride1, tl.cast(log2_scale_int & 0xFF, tl.uint8))
        tl.store(qmeta_ptr + 1 * qmeta_stride1, tl.cast((log2_scale_int >> 8) & 0xFF, tl.uint8))


def mse_scale_groups_packed(
    x_groups: torch.Tensor,
    p: torch.Tensor,
    qmeta_bytes: torch.Tensor,
    maxq: float,
    norm: float
):
    assert x_groups.is_cuda, "x_groups must be CUDA"
    assert p.is_cuda, "p must be CUDA"
    assert qmeta_bytes.is_cuda, "qmeta_bytes must be CUDA"
    assert x_groups.ndim == 2, f"Expected x_groups to be 2D, but found {x_groups.ndim} dimensions"
    n_groups, group_size = x_groups.shape
    assert group_size % 32 == 0, f"Expected group_size to be multiple of 32, but found {group_size}"
    
    P = p.shape[0]
    assert P > 0 and P <= 1024, f"P must be in (0, 1024], but got {P}"
    
    # Ensure tensors are contiguous
    x_groups = x_groups.contiguous()
    p = p.contiguous()
    qmeta_bytes = qmeta_bytes.contiguous()
    
    grid = (n_groups,)
    block_size = triton.next_power_of_2(group_size)
    assert block_size <= 1024, "Cannot use triton kernel with group size larger than 1024"
    
    # Check if L2 norm for optimization
    is_l2_norm = abs(norm - 2.0) < 1e-5
    
    mse_scale_groups_packed_triton[grid](
        x_ptr=x_groups,
        qmeta_ptr=qmeta_bytes,
        p_ptr=p,
        group_size=group_size,
        P=P,
        maxq=maxq,
        norm=norm,
        is_l2_norm=is_l2_norm,
        x_stride0=x_groups.stride(0),
        x_stride1=x_groups.stride(1),
        qmeta_stride0=qmeta_bytes.stride(0),
        qmeta_stride1=qmeta_bytes.stride(1),
        BLOCK_SIZE=block_size,
    )
    
    return qmeta_bytes

