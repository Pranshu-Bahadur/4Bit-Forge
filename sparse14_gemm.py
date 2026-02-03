import torch
import torch.nn.functional as F
from forge.backend.cuda import kernels 

from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute, moe_unpermute
)

# ---- 1) Plug your established matmul here (DO NOT TOUCH KERNEL) ----
#
# Expected:
#   qWpacked: per-expert packed weight (whatever shape you store)
#   X: bf16 [M, Cin]
# returns:
#   Y: bf16 [M, Cout]
#
def sparse14_gemm(qWpacked, X_bf16: torch.Tensor) -> torch.Tensor:
    # Replace this with your real op path:
    # e.g. return torch.ops.fourbit_forge.moe_proj_unstructured_sparse14_int4symq_gemm(qWpacked, X_bf16)
    return kernels.sparse14_gemm(qWpacked, X_bf16)


class Sparse14MoERuntime:
    def __init__(self, w13_packed, w2_packed, intermediate_size, num_experts, chunk_m=256):
        self.w13_packed = w13_packed
        self.w2_packed  = w2_packed
        self.intermediate_size = int(intermediate_size)
        self.num_experts = int(num_experts)
        self.chunk_m = int(chunk_m)

    @torch.no_grad()
    def forward(self, hidden_states, topk_ids, topk_weights):
        # hidden_states: [T,H] bf16
        T, H = hidden_states.shape
        K = topk_ids.shape[1]

        # permute
        hs_perm, _, expert_offsets, inv_idx, _ = moe_permute(
            hidden_states, None, topk_ids, n_expert=self.num_experts
        )

        # compute outputs in permuted space
        out_perm = torch.zeros((hs_perm.size(0), H), device=hidden_states.device, dtype=hidden_states.dtype)

        for e in range(self.num_experts):
            start = int(expert_offsets[e].item())
            end   = int(expert_offsets[e + 1].item())
            if end <= start:
                continue

            w13_e = self.w13_packed[e]
            w2_e  = self.w2_packed[e]

            for s in range(start, end, self.chunk_m):
                t = min(s + self.chunk_m, end)
                x = hs_perm[s:t]                          # [m,H]
                y13 = sparse14_gemm(w13_e, x)             # [m,2I]
                gate, up = y13.split(self.intermediate_size, dim=1)
                a2 = F.silu(gate) * up                    # [m,I]
                out_perm[s:t] = sparse14_gemm(w2_e, a2)   # [m,H]

        # unpermute + reduce with router weights
        out = torch.empty((T, H), device=hidden_states.device, dtype=hidden_states.dtype)
        moe_unpermute(out, out_perm, topk_weights, inv_idx, expert_first_token_offset=expert_offsets)
        return out

