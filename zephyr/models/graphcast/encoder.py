from torch import nn, Tensor


class GraphEncoder(nn.Module):
    def __init__(self, n_vars: int, hidden_size: int):
        super().__init__()
        self.embed = nn.Linear(n_vars, hidden_size)

    def forward(self, x_nodes: Tensor) -> Tensor:
        return self.embed(x_nodes)
