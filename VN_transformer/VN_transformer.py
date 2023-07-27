import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, reduce

# helper

def exists(val):
    return val is not None

# main class

class VNTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim
    ):
        super().__init__()

    def forward(
        self,
        feats,
        coors,
        mask = None
    ):
        return feats, coors
