from torch import nn

from .layers import GraphLayer


class GNNProcessor(nn.Module):
    def __init__(self, hidden_size: int, depth: int):
        super().__init__()
        self.layers = nn.ModuleList([GraphLayer(hidden_size) for _ in range(depth)])

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x
