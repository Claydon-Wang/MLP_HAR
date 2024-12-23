from random import randrange
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce
from torch.nn.modules.utils import _pair
from torchsummary import summary

def exists(val):
    return val is not None

def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) > prob_survival

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, dim_seq, attn_dim = None, causal = False, init_eps = 1e-3):
        super().__init__()
        dim_out = dim // 2
        self.causal = causal

        self.norm = nn.LayerNorm(dim_out)
        self.gate_gelu = nn.GELU()
        self.proj = nn.Conv1d(dim_seq, dim_seq, 1)

        init_eps /= dim_seq
        nn.init.uniform_(self.proj.weight, -init_eps, init_eps)
        nn.init.constant_(self.proj.bias, 1.0)

    def forward(self, x):
        device, n = x.device, x.shape[1]


        res, gate = x.chunk(2, dim = -1)
        gate = self.norm(gate)

        weight, bias = self.proj.weight, self.proj.bias

        if self.causal:
            weight, bias = weight[:n, :n], bias[:n]
            mask = torch.ones(weight.shape[:2], device = device).triu_(1).bool()
            weight = weight.masked_fill(mask[..., None], 0.)

        gate = F.conv1d(gate, weight, bias)
        out = gate * res
        # out = self.gate_gelu(out)
        return out


def MLPBlock(
    *,
    dim,
    dim_ff,
    seq_len,
    attn_dim = None,
    causal = False
):
    return nn.Sequential(
        nn.Linear(dim, dim_ff),
        nn.GELU(),
        SpatialGatingUnit(dim_ff, seq_len, attn_dim, causal),
        nn.Linear(dim_ff // 2, dim)
    )


class MLPHAR(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        ff_mult = 4,
        channels = 3,
        attn_dim = None,
        prob_survival = 1.
    ):
        super().__init__()
        image_size_h, image_size_w = _pair(image_size)
        patch_size_h, patch_size_w = _pair(patch_size)
        assert (image_size_h % patch_size_h) == 0, 'image size_h must be divisible by the patch size_h'
        assert (image_size_w % patch_size_w) == 0, 'image size_h must be divisible by the patch size_h'
        dim_ff = dim * ff_mult
        num_patches = (image_size_h // patch_size_h) * (image_size_w // patch_size_w)

        self.to_patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_size_h, p2 = patch_size_w),
            nn.Linear(channels * patch_size_h * patch_size_w, dim)
        )

        self.prob_survival = prob_survival

        self.layers = nn.ModuleList([Residual(PreNorm(dim, MLPBlock(dim = dim, dim_ff = dim_ff, seq_len = num_patches, attn_dim = attn_dim))) for i in range(depth)])

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, num_classes)
        )
        self.last_spatial_gating_output = None

    def forward(self, x):
        self.spatial_gating_outputs = []
        x = self.to_patch_embed(x)
        layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
        for layer in layers:
            x = layer(x)
            self.spatial_gating_outputs.append(x)

        x = self.to_logits(x)
        return x

def MLP_WISDM(**kwargs):
    return MLPHAR(image_size=(200, 3), patch_size=(10, 3), num_classes=6, dim=256,
                      depth=1, ff_mult=3, channels=1, attn_dim=None, prob_survival=0.99, **kwargs)

def MLP_OPPO_30(**kwargs):
    return MLPHAR(image_size=(30, 114), patch_size=(6, 19), num_classes=18, dim=256,
                      depth=1, ff_mult=4, channels=1, attn_dim=None, prob_survival=0.9, **kwargs)

def MLP_PAMAMP2(**kwargs):
    return MLPHAR(image_size=(120, 1), patch_size=(15, 1), num_classes=12, dim=256,
                      depth=1, ff_mult=3, channels=86, attn_dim=None, prob_survival=0.9, **kwargs)

def MLP_USC(**kwargs):
    return MLPHAR(image_size=(512, 6), patch_size=(64, 4), num_classes=12, dim=512,
                      depth=1, ff_mult=4, channels=1, attn_dim=None, **kwargs)


def main():
    model = MLP_WISDM().cuda()
    summary(model, (1, 200, 3))
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total_Number of params: {} |Trainable_num of params: {}'.format(total_num, trainable_num))

if __name__ == '__main__':
    main()





