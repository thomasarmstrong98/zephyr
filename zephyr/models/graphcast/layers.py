import torch
from torch import nn, Tensor
from torch_geometric.nn import MessagePassing
from typing import Tuple, Optional


class GraphLayer(MessagePassing):
    """Standard mesh GNN layer for processing on mesh nodes."""

    def __init__(self, hidden_size: int, edge_dim: int = 0):
        super().__init__(aggr='add')
        edge_input_dim = hidden_size * 2 + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, edge_index, edge_attr=None):
        return self.norm(x + self.propagate(edge_index, x=x, edge_attr=edge_attr))

    def message(self, x_i, x_j, edge_attr):
        if edge_attr is not None:
            edge_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            edge_input = torch.cat([x_i, x_j], dim=-1)
        return self.edge_mlp(edge_input)

    def update(self, aggr_out, x):
        return self.node_mlp(torch.cat([x, aggr_out], dim=-1))


class BipartiteGraphLayer(MessagePassing):
    """
    Bipartite GNN layer for message passing between two node sets.

    Used for Grid2Mesh and Mesh2Grid communication.
    """

    def __init__(
        self,
        sender_dim: int,
        receiver_dim: int,
        edge_dim: int,
        hidden_size: int,
        output_dim: int
    ):
        """
        Args:
            sender_dim: Feature dimension of sender nodes
            receiver_dim: Feature dimension of receiver nodes
            edge_dim: Feature dimension of edges
            hidden_size: Hidden dimension for MLPs
            output_dim: Output dimension
        """
        super().__init__(aggr='add', flow='source_to_target')

        # Edge MLP: processes edge features + sender/receiver node features
        self.edge_mlp = nn.Sequential(
            nn.Linear(sender_dim + receiver_dim + edge_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Node update MLP: processes receiver features + aggregated edge messages
        self.node_mlp = nn.Sequential(
            nn.Linear(receiver_dim + hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(
        self,
        x_sender: Tensor,
        x_receiver: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor
    ) -> Tensor:
        """
        Args:
            x_sender: Sender node features (N_sender, D_sender)
            x_receiver: Receiver node features (N_receiver, D_receiver)
            edge_index: Edge indices (2, E), [sender_idx, receiver_idx]
            edge_attr: Edge features (E, D_edge)

        Returns:
            Updated receiver node features (N_receiver, D_out)
        """
        return self.propagate(
            edge_index,
            x=(x_sender, x_receiver),
            edge_attr=edge_attr,
            size=(x_sender.size(0), x_receiver.size(0))
        )

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        """
        Compute edge messages.

        Args:
            x_i: Receiver node features (E, D_receiver)
            x_j: Sender node features (E, D_sender)
            edge_attr: Edge features (E, D_edge)

        Returns:
            Edge messages (E, hidden_size)
        """
        # Concatenate sender, receiver, and edge features
        edge_input = torch.cat([x_j, x_i, edge_attr], dim=-1)
        return self.edge_mlp(edge_input)

    def update(self, aggr_out: Tensor, x: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Update receiver nodes with aggregated messages.

        Args:
            aggr_out: Aggregated edge messages (N_receiver, hidden_size)
            x: Tuple of (sender features, receiver features)

        Returns:
            Updated receiver features (N_receiver, D_out)
        """
        _, x_receiver = x
        node_input = torch.cat([x_receiver, aggr_out], dim=-1)
        return self.node_mlp(node_input)
