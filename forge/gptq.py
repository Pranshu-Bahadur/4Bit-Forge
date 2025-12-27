import torch
import torch.nn as nn
from torch import Tensor
import math
import uuid

from forge.cuda import kernels 


class GPTQ(object):
    def __init__(self, 
                 layer : nn.Module,
                 bits : int = 4,
                 group_size : int = 128,
                 symmetric : bool = False,
                 quantization_scale : str = "mse",
                 quantization_order : str = "activation",
                 rel_damp : float = 1e-2,
                 algorithm : str = "babai",
                 owner = None
                 ):
        
        self.layer = layer
        self.W = self.layer.weight.clone() #(R, C) => out_features, in_features
        self.device = self.W.device
        self.H = None
        self.num_samples = torch.zeros((), device=self.device, dtype=torch.int64)
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.quantization_scale = quantization_scale
        self.quantization_order = quantization_order
        self.rel_damp = rel_damp
        self.algorithm = algorithm
        self.owner = owner if owner is not None else self

        self.perm = None
        self.pruned_ids = None
        self.prepared = False
        self.perm_inv = None

        self.id = uuid.uuid4()
    
    #for safety
    def _is_owner(self):
        return self.id == self.owner.id

    @torch.no_grad()
    def reset(self):
        self.W = self.layer.weight.clone()
        self.H = None
        self.num_samples = torch.zeros((), device=self.device, dtype=torch.int64)
        self.prepared = False
        self.perm = None
        self.pruned_ids = None
        self.perm_inv = None

        torch.cuda.empty_cache()

    @torch.no_grad()
    def update(self, X : Tensor):
        if isinstance(self.layer, nn.Linear): #Only supports Linear
            X = X.reshape(-1, X.shape[-1]).to(torch.float32) #@TODO if upcast needed? addmm might upcast to fp32
        
        new_samples = int(X.shape[0])
        if new_samples <= 0:
            return
        
        num_samples = int(self.num_samples.item()) #@TODO might cause CPU synch 
        total_samples = num_samples + new_samples
        alpha = 2.0 / total_samples

        if self.H is None:
            self.H = alpha*(X.transpose(-2, -1)@X)
        else:
            beta = num_samples / total_samples
            self.H.addmm_(X.transpose(-2, -1), X, alpha=alpha, beta=beta)
        
        self.num_samples.add_(new_samples)
    
    @torch.no_grad()
    def _prep(self):
        _owner = self.owner
        if (not self._is_owner()) and _owner.prepared and (not self.prepared):
            self.H = _owner.H.clone() #For sanilty..not **really** needed
            self.pruned_ids = _owner.pruned_ids
            self.perm = _owner.perm
            self.perm_inv = _owner.perm_inv
            self.W[:, self.pruned_ids] = 0
            self.num_samples = _owner.num_samples.clone()
            self.prepared = True
            return
        
        if _owner.H is None or (int(_owner.num_samples.item()) == 0) or torch.isnan(_owner.H).any().item():
            C = int(_owner.W.shape[-1])
            _owner.H = torch.eye(C, device=_owner.device, dtype=torch.float32)
        
        _owner.pruned_ids = (torch.diag(_owner.H) == 0)
        _owner.H[_owner.pruned_ids, : ] = 0
        _owner.H[:, _owner.pruned_ids] = 0
        _owner.H[_owner.pruned_ids, _owner.pruned_ids] = 1

    
        if _owner.quantization_order == "activation":
            _owner.perm = torch.argsort(torch.diag(_owner.H), descending=True)
        else:
            _owner.perm = torch.arange(int(_owner.W.shape[-1]), device=_owner.device)
        _owner.perm_inv = torch.argsort(_owner.perm)

        _owner.W[:, _owner.pruned_ids] = 0
        _owner.prepared = True
        _owner.H = _owner.H[_owner.perm][:,_owner.perm]

        if (not self._is_owner()) and _owner.prepared and (not self.prepared):
            self.H = _owner.H.clone() #For sanilty..not **really** needed
            self.pruned_ids = _owner.pruned_ids
            self.perm = _owner.perm
            self.perm_inv = _owner.perm_inv
            self.W[:, self.pruned_ids] = 0
            self.num_samples = _owner.num_samples.clone()
            self.prepared = True
    
    @torch.no_grad()
    def quantize(self):
        self._prep()
        scales, qzeros = self._quant_grid()
        A = self._h_factor()
        W = self.W.transpose(-2, -1)[self.perm].contiguous()
        qweight = self._solver(A, W, scales, qzeros)
        return qweight[self.perm_inv].transpose(-2, -1).contiguous(), scales, qzeros

    @torch.no_grad()
    def _h_factor(self):
        H = self.H.clone()


        zero_cols = self.W.eq(0).all(dim=0)
        if zero_cols.any():
                H[zero_cols, :] = 0
                H[:, zero_cols] = 0
                H[zero_cols, zero_cols] = 1.0

        diag = torch.diag(self.H)
        damp = float(self.rel_damp) * diag.mean()
        H.diag().add_(damp)

        try:
            if self.algorithm == "babai":
                torch.linalg.cholesky(H, upper=True, out=H)  # A where H = A^T A
            else:
                torch.linalg.cholesky(H, upper=False, out=H)     # L
                torch.cholesky_inverse(H, upper=False, out=H)    # H^{-1}
                torch.linalg.cholesky(H, upper=True, out=H)      # U where H^{-1}=U^T U
        except RuntimeError as e:
            self.issue_non_invertible = True
            #print(f"[HESSIAN] factorization failed: {e}")  # enable during bring-up
            H = torch.eye(self.W.shape[-1], device=self.device, dtype=torch.float32)

        #H.div_(H.diag()[:, None])

        return H


    @torch.no_grad()
    def _quant_grid(self, 
                    max_shrink : float = 0.2,
                    n_grid: int  = 100,
                    norm : float = 2.4
                    ):
        
        W = self.W.clone()
        R, C = W.shape
        G = (C + self.group_size - 1) // self.group_size
        pad = G * self.group_size - C

        #R, G, g_size
        if pad:
            W = torch.nn.functional(W, (0, pad))

        Wg = W.reshape(R*G, self.group_size).contiguous()

        scales, qzeros = kernels.build_group_meta_packed(
                Wg.to(torch.float32).contiguous(),
                self.bits,
                self.symmetric
                )

        if self.quantization_scale == "mse":
            p = torch.linspace(
                    1.0,
                    max_shrink,
                    n_grid,
                    dtype=torch.float32,
                    device=Wg.device
                )
            scales, qzeros = kernels.mse_scale_groups_packed(
                    Wg.to(torch.float32).contiguous(),
                    scales.contiguous(),
                    qzeros.contiguous(),
                    p,
                    float(norm),
                    self.bits,
                    self.symmetric
                )
            
        self.G = int(G)

        #scales_gr_1d = scales.view(R, G).transpose(-2, -1).contiguous().reshape(G*R)
        #qzeros_gr_1d = qzeros.view(R, G).transpose(-2, -1).contiguous().reshape(G*R)
        return scales, qzeros


    @torch.no_grad()
    def _solver(self, A, W, scales, qzeros):
        g_idx = (self.perm // self.group_size).to(torch.int32)

        if self.algorithm == 'babai':
            qw = kernels.babai_solver(
                    W.to(torch.float32),
                    A.clone(),
                    scales.clone(),
                    qzeros.clone(),
                    self.group_size,
                    self.bits,
                    self.group_size // 4,
                    g_idx,
                    self.G
                )
            return qw
        
        if self.algorithm == 'gptq':
            qw = kernels.gptq_solver(
                W.to(torch.float32),
                A.clone(),
                scales.clone(),
                qzeros.clone(),
                self.group_size,
                self.bits,
                self.group_size // 4,
                g_idx,
                self.G
            )

            return qw
