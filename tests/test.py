import pytest

import torch
from VN_transformer.VN_transformer import VNTransformer, VNInvariant, VNAttention
from VN_transformer.rotations import rot

torch.set_default_dtype(torch.float64)

# test invariant layers

def test_vn_invariant():
    layer = VNInvariant(64)

    coors = torch.randn(1, 32, 64, 3)

    R = rot(*torch.randn(3))
    out1 = layer(coors)
    out2 = layer(coors @ R)

    assert torch.allclose(out1, out2, atol = 1e-6)

# test equivariance

def test_equivariance():

    model = VNTransformer(
        dim = 64,
        depth = 2,
        dim_head = 64,
        heads = 8
    )

    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    R   = rot(*torch.randn(3))
    out1 = model(coors @ R, mask = mask)
    out2 = model(coors, mask = mask) @ R

    assert torch.allclose(out1, out2, atol = 1e-6), 'is not equivariant'

# test vn perceiver attention equivariance

def test_perceiver_vn_attention_equivariance():

    model = VNAttention(
        dim = 64,
        dim_head = 64,
        heads = 8,
        num_latents = 2
    )

    coors = torch.randn(1, 32, 64, 3)
    mask  = torch.ones(1, 32).bool()

    R   = rot(*torch.randn(3))
    out1 = model(coors @ R, mask = mask)
    out2 = model(coors, mask = mask) @ R

    assert out1.shape[1] == 2
    assert torch.allclose(out1, out2, atol = 1e-6), 'is not equivariant'

# test early fusion equivariance

def test_equivariance_with_early_fusion():

    model = VNTransformer(
        dim = 64,
        depth = 2,
        dim_head = 64,
        heads = 8,
        dim_feat = 64
    )

    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    R   = rot(*torch.randn(3))
    out1 = model(coors @ R, feats = feats, mask = mask)
    out1 = out1[..., :3]

    out2 = model(coors, feats = feats, mask = mask)
    out2 = out2[..., :3] @ R

    assert torch.allclose(out1, out2, atol = 1e-6), 'is not equivariant'
