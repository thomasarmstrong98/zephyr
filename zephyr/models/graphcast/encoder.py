from torch import nn, Tensor
from .layers import BipartiteGraphLayer


class Grid2MeshEncoder(nn.Module):
    """
    Grid2Mesh encoder using bipartite GNN.

    Embeds grid and mesh node features, then performs message passing
    from grid to mesh to create latent mesh representations.
    """

    def __init__(
        self,
        grid_node_dim: int,
        mesh_node_dim: int,
        edge_dim: int,
        hidden_size: int,
        output_size: int
    ):
        """
        Args:
            grid_node_dim: Input grid node feature dimension
            mesh_node_dim: Input mesh node feature dimension
            edge_dim: Edge feature dimension
            hidden_size: Hidden dimension for message passing
            output_size: Output latent dimension
        """
        super().__init__()

        # Embed input features to hidden size
        self.grid_node_encoder = nn.Linear(grid_node_dim, hidden_size)
        self.mesh_node_encoder = nn.Linear(mesh_node_dim, hidden_size)

        # Bipartite GNN: grid → mesh
        self.grid2mesh_gnn = BipartiteGraphLayer(
            sender_dim=hidden_size,  # grid features
            receiver_dim=hidden_size,  # mesh features
            edge_dim=edge_dim,
            hidden_size=hidden_size,
            output_dim=output_size
        )

    def forward(
        self,
        grid_features: Tensor,
        mesh_features: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            grid_features: Grid node features (N_grid, D_grid)
            mesh_features: Mesh node features (N_mesh, D_mesh)
            edge_index: Grid2mesh edge indices (2, E)
            edge_attr: Grid2mesh edge features (E, D_edge)

        Returns:
            latent_mesh: Latent mesh node features (N_mesh, output_size)
            latent_grid: Latent grid node features (N_grid, hidden_size)
        """
        # Embed node features
        latent_grid = self.grid_node_encoder(grid_features)
        latent_mesh = self.mesh_node_encoder(mesh_features)

        # Message passing: grid → mesh
        latent_mesh = self.grid2mesh_gnn(
            x_sender=latent_grid,
            x_receiver=latent_mesh,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        return latent_mesh, latent_grid
