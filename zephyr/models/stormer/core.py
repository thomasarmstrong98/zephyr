import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import scaled_dot_product_attention


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class OutputLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, n_var, img_size):
        super().__init__()

        self.n_var = n_var
        self.patch_size = patch_size
        self.img_size = img_size

        self.linear = nn.Linear(
            hidden_size, self.patch_size * self.patch_size * self.n_var, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x):
        x = self.linear(x)
        return self.unpatchify(x, self.img_size[0], self.img_size[1], self.patch_size, self.n_var)

    @staticmethod
    def unpatchify(x: torch.Tensor, h: int, w: int, patch_size: int, n_var: int):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        h_scaled = h // patch_size
        w_scaled = w // patch_size
        assert h_scaled * w_scaled == x.size(1)
        x = x.reshape(shape=(x.shape[0], h_scaled, w_scaled, patch_size, patch_size, n_var))
        x = rearrange(x, "b h w p q vars -> b vars (h p) (w q)")
        return F.interpolate(x, size=(h, w), mode="bilinear")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = torch.unbind(qkv, 2)

        x = scaled_dot_product_attention(q, k, v)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    An transformers block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden_size, int(hidden_size * mlp_ratio), hidden_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Linear(1, hidden_size)

    def forward(self, t):
        return self.net(t.unsqueeze(-1))
