from torch import nn

from .layers import GraphLayer


class GNNProcessor(nn.Module):
    def __init__(self, hidden_size: int, depth: int, edge_dim: int = 0):
        super().__init__()
        self.layers = nn.ModuleList([GraphLayer(hidden_size, edge_dim) for _ in range(depth)])

    def forward(self, x, edge_index, edge_attr=None):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x
