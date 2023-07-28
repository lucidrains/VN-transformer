import pytest

import torch
from VN_transformer.VN_transformer import VNTransformer, VNInvariant
from VN_transformer.rotations import rot

torch.set_default_dtype(torch.float64)

# test invariant layers

def test_vn_invariant():
    layer = VNInvariant(64)

    coors = torch.randn(1, 32, 64, 3)

    R = rot(*torch.randn(3))
    out1 = layer(coors)
    out2 = layer(coors @ R)
    print((out1 - out2).abs().max())
    assert torch.allclose(out1, out2, atol = 1e-6)

# test equivariance

def test_equivariance():

    model = VNTransformer(
        dim = 64,
        depth = 2,
        dim_head = 64,
        heads = 8
    )

    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    R   = rot(*torch.randn(3))
    _, out1 = model(feats, coors @ R, mask)
    out2 = model(feats, coors, mask)[1] @ R

    assert torch.allclose(out1, out2, atol = 1e-6), 'is not equivariant'
