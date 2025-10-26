from torch import nn, Tensor
from .layers import BipartiteGraphLayer


class Mesh2GridDecoder(nn.Module):
    """
    Mesh2Grid decoder using bipartite GNN.

    Performs message passing from mesh to grid to produce predictions
    on the regular lat-lon grid.
    """

    def __init__(
        self,
        mesh_dim: int,
        grid_dim: int,
        edge_dim: int,
        hidden_size: int,
        output_vars: int
    ):
        """
        Args:
            mesh_dim: Latent mesh node feature dimension
            grid_dim: Latent grid node feature dimension
            edge_dim: Edge feature dimension
            hidden_size: Hidden dimension for message passing
            output_vars: Number of output variables
        """
        super().__init__()

        # Bipartite GNN: mesh → grid
        self.mesh2grid_gnn = BipartiteGraphLayer(
            sender_dim=mesh_dim,
            receiver_dim=grid_dim,
            edge_dim=edge_dim,
            hidden_size=hidden_size,
            output_dim=hidden_size
        )

        # Project to output variables
        self.output_proj = nn.Linear(hidden_size, output_vars)

    def forward(
        self,
        mesh_features: Tensor,
        grid_features: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor
    ) -> Tensor:
        """
        Args:
            mesh_features: Latent mesh node features (N_mesh, D_mesh)
            grid_features: Latent grid node features (N_grid, D_grid)
            edge_index: Mesh2grid edge indices (2, E)
            edge_attr: Mesh2grid edge features (E, D_edge)

        Returns:
            Grid predictions (N_grid, output_vars)
        """
        # Message passing: mesh → grid
        grid_output = self.mesh2grid_gnn(
            x_sender=mesh_features,
            x_receiver=grid_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        # Project to output variables
        return self.output_proj(grid_output)
