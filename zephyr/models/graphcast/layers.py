import torch
from torch import nn
from torch_geometric.nn import MessagePassing


class GraphLayer(MessagePassing):
    def __init__(self, hidden_size: int):
        super().__init__(aggr='add')
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, edge_index):
        return self.norm(x + self.propagate(edge_index, x=x))

    def message(self, x_i, x_j):
        return self.edge_mlp(torch.cat([x_i, x_j], dim=-1))

    def update(self, aggr_out, x):
        return self.node_mlp(torch.cat([x, aggr_out], dim=-1))
