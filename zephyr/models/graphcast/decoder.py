from torch import nn, Tensor


class GraphDecoder(nn.Module):
    def __init__(self, hidden_size: int, n_vars: int):
        super().__init__()
        self.project = nn.Linear(hidden_size, n_vars)

    def forward(self, x_nodes: Tensor) -> Tensor:
        return self.project(x_nodes)
