<img src="./vn-transformer.png" width="300px"></img>

## VN (Vector Neuron) Transformer

A <a href="https://arxiv.org/abs/2206.04176">Transformer made of Rotation-equivariant Attention</a> using <a href="https://arxiv.org/abs/2104.12229">Vector Neurons</a>

<a href="https://openreview.net/forum?id=EiX2L4sDPG">Open Review</a>

## Appreciation

- <a href="https://stability.ai/">StabilityAI</a> for the generous sponsorship, as well as my other sponsors, for affording me the independence to open source artificial intelligence.

## Install

```bash
$ pip install VN-transformer
```

## Usage

```python
import torch
from VN_transformer import VNTransformer

model = VNTransformer(
    dim = 64,
    depth = 2,
    dim_head = 64,
    heads = 8,
    dim_feat = 64,       # will default to early fusion, since this was the best performing
    bias_epsilon = 1e-6  # in this paper, they propose breaking equivariance with a tiny bit of bias noise in the VN linear. they claim this leads to improved stability. setting this to 0 would turn off the epsilon approximate equivariance
)

coors = torch.randn(1, 32, 3)    # (batch, sequence, spatial coordinates)
feats = torch.randn(1, 32, 64)

coors_out, feats_out = model(coors, feats = feats) # (1, 32, 3), (1, 32, 64)
```

## Tests

Confidence in equivariance

```bash
$ python setup.py test
```

## Example

First install `sidechainnet`

```bash
$ pip install sidechainnet
```

Then run the protein backbone denoising task

```bash
$ python denoise.py
```

It does not perform as well as <a href="https://github.com/lucidrains/En-transformer">En-Transformer</a>, nor <a href="https://github.com/lucidrains/equiformer-pytorch">Equiformer</a>

## Citations

```bibtex
@inproceedings{Assaad2022VNTransformerRA,
    title   = {VN-Transformer: Rotation-Equivariant Attention for Vector Neurons},
    author  = {Serge Assaad and C. Downey and Rami Al-Rfou and Nigamaa Nayakanti and Benjamin Sapp},
    year    = {2022}
}
```

```bibtex
@article{Deng2021VectorNA,
    title   = {Vector Neurons: A General Framework for SO(3)-Equivariant Networks},
    author  = {Congyue Deng and Or Litany and Yueqi Duan and Adrien Poulenard and Andrea Tagliasacchi and Leonidas J. Guibas},
    journal = {2021 IEEE/CVF International Conference on Computer Vision (ICCV)},
    year    = {2021},
    pages   = {12180-12189},
    url     = {https://api.semanticscholar.org/CorpusID:233394028}
}
```

```bibtex
@inproceedings{Kim2020TheLC,
    title   = {The Lipschitz Constant of Self-Attention},
    author  = {Hyunjik Kim and George Papamakarios and Andriy Mnih},
    booktitle = {International Conference on Machine Learning},
    year    = {2020},
    url     = {https://api.semanticscholar.org/CorpusID:219530837}
}
```

```bibtex
@inproceedings{dao2022flashattention,
    title   = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
    author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2022}
}
```