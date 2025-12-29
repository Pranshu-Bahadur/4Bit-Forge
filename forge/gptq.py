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
                 algorithm : str = "gptq", #@TODO Note babai is a WIP
                 owner = None,
                 device = None
                 ):
        
        self.layer = layer
        self.W = self.layer.weight.clone() #(R, C) => out_features, in_features
        self.device = device
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
        if self._is_owner():
            if isinstance(self.owner.layer, nn.Linear): #Only supports Linear
                X = X.reshape(-1, X.shape[-1]).to(device=self.owner.device, dtype=torch.float32) #@TODO if upcast needed? addmm might upcast to fp32
            
            new_samples = int(X.shape[0])

            if new_samples == 0:
                return
    
            num_samples = int(self.owner.num_samples.item()) #@TODO might cause CPU synch 
            total_samples = num_samples + new_samples
            alpha = 2.0 / total_samples

            if self.owner.H is None:
                self.owner.H = X.T@(alpha*X)

            beta = num_samples / total_samples
            self.owner.H .addmm_(X.transpose(-2, -1), X, alpha=alpha, beta=beta)
            self.owner.num_samples.add_(new_samples)
    
    @torch.no_grad()
    def _prep(self):
        if self._is_owner() and self.prepared:
            return
        if (not self._is_owner()) and self.owner.prepared and (not self.prepared):
            self.H = self.owner.H.clone() #For sanilty..not **really** needed
            self.pruned_ids = self.owner.pruned_ids.clone()
            self.perm = self.owner.perm.clone()
            self.perm_inv = self.owner.perm_inv.clone()
            self.W[:, self.pruned_ids] = 0
            self.num_samples = self.owner.num_samples.clone()
            self.prepared = True
            return
        
        if self.owner.H is None or (int(self.owner.num_samples.item()) == 0) or torch.isnan(self.owner.H).any().item():
            C = int(self.owner.W.shape[-1])
            self.owner.H = torch.eye(C, device=self.owner.device, dtype=torch.float32)
        
        self.owner.pruned_ids = (torch.diag(self.owner.H) == 0)
        self.owner.H[self.owner.pruned_ids, : ] = 0
        self.owner.H[:, self.owner.pruned_ids] = 0
        self.owner.H[self.owner.pruned_ids, self.owner.pruned_ids] = 1

    
        if self.owner.quantization_order == "activation": 
            if self.owner.algorithm == 'babai':
                self.owner.perm = torch.argsort(self.owner.H.diag().to(device=self.owner.device), descending=False).to(device=self.owner.device) #Babai is back to front -> ascending order
            else:
                self.owner.perm = torch.argsort(self.owner.H.diag().to(device=self.owner.device), descending=True).to(device=self.owner.device)
        else:
            self.owner.perm = torch.arange(int(self.owner.W.shape[-1]), device=self.owner.device)
        self.owner.perm_inv = torch.argsort(self.owner.perm).to(device=self.owner.device)

        self.owner.W[:, self.owner.pruned_ids] = 0
        self.owner.prepared = True
        self.owner.H = self.owner.H[self.owner.perm][:,self.owner.perm]

        if (not self._is_owner()) and self.owner.prepared and (not self.prepared):
            self.H = self.owner.H.clone() #For sanilty..not **really** needed
            self.pruned_ids = self.owner.pruned_ids.clone()
            self.perm = self.owner.perm.clone()
            self.perm_inv = self.owner.perm_inv.clone()
            self.W[:, self.pruned_ids] = 0
            self.num_samples = self.owner.num_samples.clone()
            self.prepared = True
    
    @torch.no_grad()
    def quantize(self):
        self._prep()
        scales, qzeros = self._quant_grid()
        A = self._h_factor()
        W = self.W.clone().transpose(-2, -1)[self.perm]#.contiguous()
        qweight = self._solver(A, W, scales, qzeros)
        return qweight[self.perm_inv].transpose(-2, -1).contiguous(), scales, qzeros

    @torch.no_grad()
    def _h_factor(self):
        H = self.H#.clone()

        zero_cols = self.W.clone().eq(0).all(dim=0)
        if zero_cols.any():
            H[zero_cols, :] = 0
            H[:, zero_cols] = 0
            H[zero_cols, zero_cols] = 1.0

        
        diag = torch.diag(H)
        damp = float(self.rel_damp) * diag.mean()
        H[range(self.W.shape[-1]), range(self.W.shape[-1])] += damp
        

        if self.algorithm == "babai":
            H, info = torch.linalg.cholesky_ex(H, upper=True)  # A where H = A^T A
        else:
            H, info = torch.linalg.cholesky_ex(H, upper=False)
            H = torch.cholesky_inverse(H, upper=False)    # H^{-1}
            H, info2 = torch.linalg.cholesky_ex(H, upper=True)
            info.add_(info2)
            H.div_(H.diag()[:, None])

        if info.item() > 0:
            self.issue_non_invertible = True
            print(f"[HESSIAN] factorization failed at {self.layer.name}")  # enable during bring-up
            H = torch.eye(self.W.shape[-1], device=self.device, dtype=torch.float32)
        
        return H#.to(torch.float32)


    @torch.no_grad()
    def _quant_grid(self, 
                    max_shrink : float = 0.2,
                    n_grid: int  = 100,
                    norm : float = 2.4
                    ):
        
        W = self.W.clone()
        R, C = W.shape
        #@TODO FIX PADDING LOGIC
        G = (C + self.group_size - 1) // self.group_size
        pad = (G * self.group_size) - C

        #R, G, g_size
        if pad:
            W = torch.nn.functional.pad(W, (0, pad))

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

        #scales = scales.view(R, G).transpose(-2, -1).clone().reshape(G*R).contiguous()
        #qzeros = qzeros.view(R, G).transpose(-2, -1).clone().reshape(G*R).contiguous()
        return scales, qzeros


    @torch.no_grad()
    def _solver(self, A, W, scales, qzeros):
        g_idx = (self.perm // self.group_size).to(torch.int32)

        if self.algorithm == 'babai': #@TODO Fix babai kernel
            qw = kernels.babai_solver(
                    W.to(torch.float32),
                    A, #.clone(),
                    scales, #.clone(),
                    qzeros, #.clone(),
                    self.group_size,
                    self.bits,
                    self.group_size // 4,
                    g_idx,
                    self.G
                )
            return qw
        
        if self.algorithm == 'gptq':
            C, R = W.shape

            #scales = scales.clone().view(R, self.G).repeat_interleave(self.group_size, dim=1)[:, :C].transpose(-2, -1)[self.perm]   # (C, R) fp32
            #qzeros = qzeros.clone().view(R, self.G).repeat_interleave(self.group_size, dim=1)[:, :C].transpose(-2, -1)[self.perm]
            qw = kernels.gptq_solver(
                W.to(torch.float32),
                A,
                scales,
                qzeros,
                g_idx,
                self.G,
                self.group_size,
                self.bits
            )

            return qw
