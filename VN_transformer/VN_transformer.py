import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# helper

def exists(val):
    return val is not None

def inner_dot_product(x, y, *, dim = -1, keepdim = True):
    return (x * y).sum(dim = dim, keepdim = keepdim)

# equivariant modules

class VNLinear(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim_out, dim_in))

    def forward(self, x):
        return einsum('... i c, o i -> ... o c', x, self.weight)

class VNRelu(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim))
        self.U = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x):
        q = einsum('... i c, o i -> ... o c', x, self.W)
        k = einsum('... i c, o i -> ... o c', x, self.U)

        qk = inner_dot_product(q, k)

        normed_k = F.normalize(k, dim = -1)
        q_projected_on_k = q - inner_dot_product(q, normed_k) * normed_k

        out = torch.where(
            qk >= 0.,
            q,
            q_projected_on_k
        )

        return out

# main class

class VNTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        reduce_dim_out = True
    ):
        super().__init__()

        self.vn_proj_in = nn.Sequential(
            Rearrange('... c -> ... 1 c'),
            VNLinear(1, dim)
        )

        self.act = VNRelu(dim)

        if reduce_dim_out:
            self.vn_proj_out = nn.Sequential(
                VNLinear(dim, 1),
                Rearrange('... 1 c -> ... c')
            )
        else:
            self.vn_proj_out = nn.Identity()

    def forward(
        self,
        feats,
        coors,
        mask = None
    ):
        coors = self.vn_proj_in(coors)
        coors = self.act(coors)
        coors = self.vn_proj_out(coors)
        return feats, coors
