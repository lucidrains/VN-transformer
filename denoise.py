import torch
import torch.nn.functional as F
from torch.optim import Adam

from einops import rearrange, repeat

import sidechainnet as scn
from VN_transformer import VNTransformer

BATCH_SIZE = 1
GRADIENT_ACCUMULATE_EVERY = 16
MAX_SEQ_LEN = 256
DEFAULT_TYPE = torch.float64

torch.set_default_dtype(DEFAULT_TYPE)

def cycle(loader, len_thres = MAX_SEQ_LEN):
    while True:
        for data in loader:
            if data.seqs.shape[1] > len_thres:
                continue
            yield data

transformer = VNTransformer(
    num_tokens = 24,
    dim = 64,
    depth = 4,
    dim_head = 64,
    heads = 8,
    dim_feat = 64,
    bias_epsilon = 1e-6,
    l2_dist_attn = True,
    flash_attn = False
).cuda()

data = scn.load(
    casp_version = 12,
    thinning = 30,
    with_pytorch = 'dataloaders',
    batch_size = BATCH_SIZE,
    dynamic_batching = False    
)

# Add gaussian noise to the coords
# Testing the refinement algorithm

dl = cycle(data['train'])
optim = Adam(transformer.parameters(), lr = 1e-4)

for _ in range(10000):
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        batch = next(dl)
        seqs, coords, masks = batch.seqs, batch.crds, batch.msks

        seqs = seqs.cuda().argmax(dim = -1)
        coords = coords.cuda().type(torch.get_default_dtype())
        masks = masks.cuda().bool()

        l = seqs.shape[1]
        coords = rearrange(coords, 'b (l s) c -> b l s c', s = 14)

        # Keeping only the backbone coordinates
        coords = coords[:, :, 0:3, :]
        coords = rearrange(coords, 'b l s c -> b (l s) c')

        seq = repeat(seqs, 'b n -> b (n c)', c = 3)
        masks = repeat(masks, 'b n -> b (n c)', c = 3)

        noised_coords = coords + torch.randn_like(coords).cuda()

        type1_out, _ = transformer(
            noised_coords,
            feats = seq,
            mask = masks
        )

        denoised_coords = noised_coords + type1_out

        loss = F.mse_loss(denoised_coords[masks], coords[masks]) 
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    print('loss:', loss.item())
    optim.step()
    optim.zero_grad()
