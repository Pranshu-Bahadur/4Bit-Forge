import torch
import torch.nn.functional as F
from forge.backend.cuda import kernels 


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
    """
    Option B:
      - w13_packed: stacked [gate_proj; up_proj] per expert
      - w2_packed: down_proj per expert
    Routing:
      - topk_ids: [T, topk] int64/int32
      - topk_weights: [T, topk] fp16/bf16/fp32

    This implementation:
      1) permutes tokens by expert id (grouped contiguous)
      2) for each expert group, runs:
           y13 = gemm(w13, x) -> split -> silu(gate)*up
           y2  = gemm(w2, a2)
      3) unpermutes back to original (token, k) order and reduces with topk_weights.
    """

    def __init__(
        self,
        w13_packed: torch.Tensor,   # [E, ...] container of per-expert packed weights
        w2_packed: torch.Tensor,    # [E, ...]
        intermediate_size: int,     # e.g. 7168
        num_experts: int,           # e.g. 128
        *,
        chunk_m: int = 256,         # peak-memory control inside each expert group
    ):
        self.w13_packed = w13_packed
        self.w2_packed = w2_packed
        self.intermediate_size = int(intermediate_size)
        self.num_experts = int(num_experts)
        self.chunk_m = int(chunk_m)

        # Optional: try vLLM CUDA permute/unpermute if available.
        self._vllm_moe = None
        try:
            from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (  # type: ignore
                moe_permute,
                moe_unpermute,
            )
            self._vllm_moe = (moe_permute, moe_unpermute)
        except Exception:
            self._vllm_moe = None

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,   # [T, H] bf16
        topk_ids: torch.Tensor,        # [T, K] int
        topk_weights: torch.Tensor,    # [T, K] float
    ) -> torch.Tensor:
        assert hidden_states.dtype == torch.bfloat16, "expected bf16 hidden_states"
        T, H = hidden_states.shape
        K = topk_ids.shape[1]
        assert topk_ids.shape == (T, K)
        assert topk_weights.shape == (T, K)

        # ------------------------------------------------------------
        # 1) PERMUTE: group tokens by expert (so each expert gets a contiguous slice)
        # ------------------------------------------------------------
        if self._vllm_moe is not None:
            # If vLLM’s moe_permute/moe_unpermute API matches, you can use it.
            # BUT: the signature has changed across versions. So we keep the
            # pure-PyTorch fallback as the guaranteed-correct path.
            pass

        # Pure PyTorch correct fallback:
        flat_ids = topk_ids.reshape(-1).to(torch.int64)                # [T*K]
        flat_w = topk_weights.reshape(-1).to(torch.float32)            # [T*K] (reduce will be fp32)
        # Repeat hidden states for each top-k choice
        hs_rep = hidden_states.repeat_interleave(K, dim=0)             # [T*K, H]

        # Stable sort by expert id -> contiguous groups per expert
        sorted_ids, perm = flat_ids.sort(stable=True)                  # perm maps (unsorted)->(sorted)
        hs_perm = hs_rep.index_select(0, perm)                         # [T*K, H] grouped by expert

        # Build inverse permutation to restore original (token,k) order later
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)

        # Count tokens per expert and prefix sum offsets
        # expert_num_tokens[e] = how many (token,k) pairs routed to expert e in this batch
        expert_num_tokens = torch.bincount(sorted_ids, minlength=self.num_experts).to(torch.int64)  # [E]
        expert_offsets = torch.empty(self.num_experts + 1, device=hidden_states.device, dtype=torch.int64)
        expert_offsets[0] = 0
        expert_offsets[1:] = torch.cumsum(expert_num_tokens, dim=0)

        # ------------------------------------------------------------
        # 2) EXPERTS: run w13 -> silu*mul -> w2 for each expert slice
        # ------------------------------------------------------------
        out_perm = torch.empty((T * K, H), device=hidden_states.device, dtype=torch.bfloat16)

        for e in range(self.num_experts):
            start = int(expert_offsets[e].item())
            end = int(expert_offsets[e + 1].item())
            if end <= start:
                continue

            w13_e = self.w13_packed[e]
            w2_e = self.w2_packed[e]

            # Chunk within expert slice to cap peak activation memory:
            # y13 chunk is [m, 2*I], a2 is [m, I]
            # This avoids materializing [T*K, 2*I] for big prefills.
            for s in range(start, end, self.chunk_m):
                t = min(s + self.chunk_m, end)
                x = hs_perm[s:t]                                       # [m, H] bf16

                y13 = sparse14_gemm(w13_e, x)                          # [m, 2I] bf16
                gate, up = y13.split(self.intermediate_size, dim=1)    # each [m, I]
                a2 = F.silu(gate) * up                                 # [m, I] bf16 (torch will keep bf16)

                y2 = sparse14_gemm(w2_e, a2)                           # [m, H] bf16
                out_perm[s:t].copy_(y2)

        # ------------------------------------------------------------
        # 3) UNPERMUTE + REDUCE: restore original (token,k) order then sum_k w * out
        # ------------------------------------------------------------
        out_pairs = out_perm.index_select(0, inv_perm)                 # [T*K, H] in original pair order
        out_pairs = out_pairs.view(T, K, H).to(torch.float32)          # accumulate in fp32
        out = (out_pairs * flat_w.view(T, K, 1)).sum(dim=1)            # [T, H] fp32
        return out.to(torch.bfloat16)
