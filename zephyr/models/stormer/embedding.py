import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn.init import trunc_normal_


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class WeatherEmbedding(nn.Module):
    def __init__(self, img_size, n_vars, embed_dim, patch_size, num_heads):
        super().__init__()

        self.img_size = img_size
        self.n_vars = n_vars
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.embed = nn.Conv2d(
            self.n_vars,
            self.n_vars * self.embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
            groups=self.n_vars,
        )
        self.var_embed = nn.Parameter(
            torch.zeros(1, self.n_vars, self.embed_dim), requires_grad=True
        )

        # variable aggregation: a learnable query and a single-layer cross attention
        self.channel_query = nn.Parameter(torch.ones(1, 1, self.embed_dim), requires_grad=True)
        self.channel_agg = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True)

        # positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.embed_dim), requires_grad=True
        )

        self.init_weights()

    def init_weights(self):
        # initialize pos_emb and var_emb
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        var_embed = get_1d_sincos_pos_embed_from_grid(
            self.var_embed.shape[-1], np.arange(self.n_vars)
        )
        self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

        def _init_weights(m: nn.Module):
            """initialize nn.Linear and nn.LayerNorm"""
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, x):
        b, v, h, w = x.shape
        x = self.embed(x)
        x = rearrange(x, "b (v e) h w -> b v (h w) e", v=self.n_vars)
        x = x + self.var_embed.unsqueeze(2)
        x = x + self.pos_embed.unsqueeze(1)
        x = rearrange(x, "b v n e -> (b n) v e")
        x, _ = self.channel_agg(self.channel_query.repeat_interleave(x.size(0), dim=0), x, x)
        x = rearrange(x, "(b n) 1 e -> b n e", b=b)
        return x
