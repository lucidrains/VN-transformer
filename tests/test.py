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

@pytest.mark.parametrize('l2_dist_attn', [True, False])
def test_equivariance(l2_dist_attn):

    model = VNTransformer(
        dim = 64,
        depth = 2,
        dim_head = 64,
        heads = 8,
        l2_dist_attn = l2_dist_attn
    )

    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    R   = rot(*torch.randn(3))
    out1 = model(coors @ R, mask = mask)
    out2 = model(coors, mask = mask) @ R

    assert torch.allclose(out1, out2, atol = 1e-6), 'is not equivariant'

# test vn perceiver attention equivariance

@pytest.mark.parametrize('l2_dist_attn', [True, False])
def test_perceiver_vn_attention_equivariance(l2_dist_attn):

    model = VNAttention(
        dim = 64,
        dim_head = 64,
        heads = 8,
        num_latents = 2,
        l2_dist_attn = l2_dist_attn
    )

    coors = torch.randn(1, 32, 64, 3)
    mask  = torch.ones(1, 32).bool()

    R   = rot(*torch.randn(3))
    out1 = model(coors @ R, mask = mask)
    out2 = model(coors, mask = mask) @ R

    assert out1.shape[1] == 2
    assert torch.allclose(out1, out2, atol = 1e-6), 'is not equivariant'

# test SO(3) early fusion equivariance

@pytest.mark.parametrize('l2_dist_attn', [True, False])
def test_equivariance_with_early_fusion(l2_dist_attn):

    model = VNTransformer(
        dim = 64,
        depth = 2,
        dim_head = 64,
        heads = 8,
        dim_feat = 64,
        l2_dist_attn = l2_dist_attn
    )

    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    R   = rot(*torch.randn(3))
    out1, _ = model(coors @ R, feats = feats, mask = mask, return_concatted_coors_and_feats = False)

    out2, _ = model(coors, feats = feats, mask = mask, return_concatted_coors_and_feats = False)
    out2 = out2 @ R

    assert torch.allclose(out1, out2, atol = 1e-6), 'is not equivariant'

# test SE(3) early fusion equivariance

@pytest.mark.parametrize('l2_dist_attn', [True, False])
def test_se3_equivariance_with_early_fusion(l2_dist_attn):

    model = VNTransformer(
        dim = 64,
        depth = 2,
        dim_head = 64,
        heads = 8,
        dim_feat = 64,
        translation_equivariance = True,
        l2_dist_attn = l2_dist_attn
    )

    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    T   = torch.randn(3)
    R   = rot(*torch.randn(3))
    out1, _ = model((coors + T) @ R, feats = feats, mask = mask, return_concatted_coors_and_feats = False)

    out2, _ = model(coors, feats = feats, mask = mask, return_concatted_coors_and_feats = False)
    out2 = (out2 + T) @ R

    assert torch.allclose(out1, out2, atol = 1e-6), 'is not equivariant'
