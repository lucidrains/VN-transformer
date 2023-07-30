import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

# helper

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def inner_dot_product(x, y, *, dim = -1, keepdim = True):
    return (x * y).sum(dim = dim, keepdim = keepdim)

# layernorm

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# equivariant modules

class VNLinear(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        bias_epsilon = 0.
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim_out, dim_in))

        self.bias = None
        self.bias_epsilon = bias_epsilon

        # in this paper, they propose going for quasi-equivariance with a small bias, controllable with epsilon, which they claim lead to better stability and results

        if bias_epsilon > 0.:
            self.bias = nn.Parameter(torch.randn(dim_out))

    def forward(self, x):
        out = einsum('... i c, o i -> ... o c', x, self.weight)

        if exists(self.bias):
            bias = F.normalize(self.bias, dim = -1) * self.bias_epsilon
            out = out + rearrange(bias, '... -> ... 1')

        return out

class VNReLU(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.W = nn.Parameter(torch.randn(dim, dim))
        self.U = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x):
        q = einsum('... i c, o i -> ... o c', x, self.W)
        k = einsum('... i c, o i -> ... o c', x, self.U)

        qk = inner_dot_product(q, k)

        k_norm = k.norm(dim = -1, keepdim = True).clamp(min = self.eps)
        q_projected_on_k = q - inner_dot_product(q, k / k_norm) * k

        out = torch.where(
            qk >= 0.,
            q,
            q_projected_on_k
        )

        return out

class VNAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dim_coor = 3,
        bias_epsilon = 0.,
        l2_dist_attn = False,
        num_latents = None   # setting this would enable perceiver-like cross attention from latents to sequence, with the latents derived from VNWeightedPool
    ):
        super().__init__()
        self.scale = (dim_coor * dim_head) ** -0.5
        dim_inner = dim_head * heads
        self.heads = heads

        self.to_q_input = None
        if exists(num_latents):
            self.to_q_input = VNWeightedPool(dim, num_pooled_tokens = num_latents, squeeze_out_pooled_dim = False)

        self.to_q = VNLinear(dim, dim_inner, bias_epsilon = bias_epsilon)
        self.to_kv = VNLinear(dim, dim_inner * 2, bias_epsilon = bias_epsilon)
        self.to_out = VNLinear(dim_inner, dim, bias_epsilon = bias_epsilon)

        self.l2_dist_attn = l2_dist_attn

        # use l2 distance squared for attention
        # this was used in the invariant point attention in alphafold2, vitgan, gigagan, and also analyzed in depth https://arxiv.org/abs/2006.04710
        # have also tested this in equiformer variant https://github.com/lucidrains/equiformer-pytorch, where i found it to converge much nicer given the fit with spatial coordinates

    def forward(self, x, mask = None):
        """
        einstein notation
        b - batch
        n - sequence
        h - heads
        d - feature dimension (channels)
        c - coordinate dimension (3 for 3d space)
        i - source sequence dimension
        j - target sequence dimension
        """

        c = x.shape[-1]

        if exists(self.to_q_input):
            q_input = self.to_q_input(x, mask = mask)
        else:
            q_input = x

        q, k, v = (self.to_q(q_input), *self.to_kv(x).chunk(2, dim = -2))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) c -> b h n d c', h = self.heads), (q, k, v))

        sim = einsum('b h i d c, b h j d c -> b h i j', q, k)

        if self.l2_dist_attn:
            # -cdist squared == (-q^2 + 2qk - k^2)
            # so simply work off the qk above
            q_squared = reduce(q ** 2, 'b h i d c -> b h i 1', 'sum')
            k_squared = reduce(k ** 2, 'b h j d c -> b h 1 j', 'sum')
            sim = sim * 2 - q_squared - k_squared

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        sim = sim * self.scale

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d c -> b h i d c', attn, v)

        out = rearrange(out, 'b h n d c -> b n (h d) c')
        return self.to_out(out)

def VNFeedForward(dim, mult = 4, bias_epsilon = 0.):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        VNLinear(dim, dim_inner, bias_epsilon = bias_epsilon),
        VNReLU(dim_inner),
        VNLinear(dim_inner, dim, bias_epsilon = bias_epsilon)
    )

class VNLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.ln = LayerNorm(dim)

    def forward(self, x):
        norms = x.norm(dim = -1)
        x = x / rearrange(norms.clamp(min = self.eps), '... -> ... 1')
        ln_out = self.ln(norms)
        return x * rearrange(ln_out, '... -> ... 1')

class VNWeightedPool(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        num_pooled_tokens = 1,
        squeeze_out_pooled_dim = True
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.weight = nn.Parameter(torch.randn(num_pooled_tokens, dim, dim_out))
        self.squeeze_out_pooled_dim = num_pooled_tokens == 1 and squeeze_out_pooled_dim

    def forward(self, x, mask = None):
        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1 1')
            x = x.masked_fill(~mask, 0.)
            numer = reduce(x, 'b n d c -> b d c', 'sum')
            denom = mask.sum(dim = 1)
            mean_pooled = numer / denom.clamp(min = 1e-6)
        else:
            mean_pooled = reduce(x, 'b n d c -> b d c', 'mean')

        out = einsum('b d c, m d e -> b m e c', mean_pooled, self.weight)

        if not self.squeeze_out_pooled_dim:
            return out

        out = rearrange(out, 'b 1 d c -> b d c')
        return out

# equivariant VN transformer encoder

class VNTransformerEncoder(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        dim_coor = 3,
        ff_mult = 4,
        final_norm = False,
        bias_epsilon = 0.,
        l2_dist_attn = False
    ):
        super().__init__()
        self.dim = dim
        self.dim_coor = dim_coor

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                VNAttention(dim = dim, dim_head = dim_head, heads = heads, bias_epsilon = bias_epsilon, l2_dist_attn = l2_dist_attn),
                VNLayerNorm(dim),
                VNFeedForward(dim = dim, mult = ff_mult, bias_epsilon = bias_epsilon),
                VNLayerNorm(dim)
            ]))

        self.norm = VNLayerNorm(dim) if final_norm else nn.Identity()

    def forward(
        self,
        x,
        mask = None
    ):
        *_, d, c = x.shape

        assert x.ndim == 4 and d == self.dim and c == self.dim_coor, 'input needs to be in the shape of (batch, seq, dim ({self.dim}), coordinate dim ({self.dim_coor}))'

        for attn, attn_post_ln, ff, ff_post_ln in self.layers:
            x = attn_post_ln(attn(x, mask = mask)) + x
            x = ff_post_ln(ff(x)) + x

        return self.norm(x)

# invariant layers

class VNInvariant(nn.Module):
    def __init__(
        self,
        dim,
        dim_coor = 3,

    ):
        super().__init__()
        self.mlp = nn.Sequential(
            VNLinear(dim, dim_coor),
            VNReLU(dim_coor),
            Rearrange('... d e -> ... e d')
        )

    def forward(self, x):
        return einsum('b n d i, b n i o -> b n o', x, self.mlp(x))

# main class

class VNTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_feat = None,
        dim_head = 64,
        heads = 8,
        dim_coor = 3,
        reduce_dim_out = True,
        bias_epsilon = 0.,
        l2_dist_attn = True
    ):
        super().__init__()
        dim_feat = default(dim_feat, 0)
        self.dim_feat = dim_feat
        self.dim_coor_total = dim_coor + dim_feat

        self.vn_proj_in = nn.Sequential(
            Rearrange('... c -> ... 1 c'),
            VNLinear(1, dim, bias_epsilon = bias_epsilon)
        )

        self.encoder = VNTransformerEncoder(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            bias_epsilon = bias_epsilon,
            dim_coor = self.dim_coor_total ,
            l2_dist_attn = l2_dist_attn
        )

        if reduce_dim_out:
            self.vn_proj_out = nn.Sequential(
                VNLayerNorm(dim),
                VNLinear(dim, 1, bias_epsilon = bias_epsilon),
                Rearrange('... 1 c -> ... c')
            )
        else:
            self.vn_proj_out = nn.Identity()

    def forward(
        self,
        coors,
        *,
        feats = None,
        mask = None
    ):
        if exists(feats):
            assert feats.ndim == 3
            assert feats.shape[-1] == self.dim_feat, f'dim_feat should be set to {feats.shape[-1]}'
            coors = torch.cat((coors, feats), dim = -1)

        assert coors.shape[-1] == self.dim_coor_total

        coors = self.vn_proj_in(coors)
        coors = self.encoder(coors, mask = mask)
        coors = self.vn_proj_out(coors)
        return coors
