import pytest

import torch
from VN_transformer.VN_transformer import VNTransformer
from VN_transformer.rotations import rot

# test equivariance

def test_equivariance():

    model = VNTransformer(
        dim = 64
    )

    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    R   = rot(*torch.randn(3))
    _, out1 = model(feats, coors @ R, mask)
    out2 = model(feats, coors, mask)[1] @ R

    assert torch.allclose(out1, out2, atol = 1e-4), 'is not equivariant'
