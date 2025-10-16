import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class GraphAttention(MessagePassing):
    """Graph attention mechanism using message passing."""

    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__(aggr='add', node_dim=0)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x, edge_index):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        out = self.propagate(edge_index, q=q, k=k, v=v, size=None)
        out = out.reshape(-1, self.hidden_size)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def message(self, q_i, k_j, v_j, edge_index, size):
        attn = (q_i * k_j).sum(dim=-1) / (self.head_dim ** 0.5)
        attn = softmax(attn, edge_index[1], num_nodes=size[1])
        return v_j * attn.unsqueeze(-1)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class GraphBlock(nn.Module):
    """Transformer block with graph attention."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = GraphAttention(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, edge_index, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), edge_index)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class GraphProcessor(nn.Module):
    """Graph-based processor for Stormer."""

    def __init__(self, hidden_size: int, depth: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            GraphBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])

    def forward(self, x, edge_index, forecast_timedelta):
        for block in self.blocks:
            x = block(x, edge_index, forecast_timedelta)
        return x
