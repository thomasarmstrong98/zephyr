import torch
from torch import nn, Tensor
from typing import Dict, List, Optional, Tuple
import numpy as np

from .encoder import Grid2MeshEncoder
from .processor import GNNProcessor
from .decoder import Mesh2GridDecoder
from .spatial_features import (
    compute_grid_node_features,
    compute_mesh_node_features,
    compute_bipartite_edge_features,
    compute_mesh_edge_features
)
from ..base import GraphWeatherModel
from ..graphs import create_icosahedral_graph
from ...data.structures import WeatherBatch


class GraphCast(nn.Module, GraphWeatherModel):
    def __init__(
        self,
        img_size: Tuple[int, int],
        variables: List[str],
        mesh_levels: int,
        hidden_size: int,
        depth: int,
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.variables = variables
        self.img_size = img_size
        self.hidden_size = hidden_size

        # Create icosahedral mesh and bipartite connectivity
        self.graph_data = create_icosahedral_graph(mesh_levels, img_size)

        # Register mesh edges as buffer
        self.register_buffer('mesh_edge_index', self.graph_data.edge_index)
        self.register_buffer('grid2mesh_edge_index', self.graph_data.grid2mesh_edge_index)
        self.register_buffer('mesh2grid_edge_index', self.graph_data.mesh2grid_edge_index)

        # Precompute and register static spatial features
        self._precompute_spatial_features()

        # Grid node features: input variables
        # Mesh node features: spatial features (6D)
        # Edge features: relative position + distance (4D)
        grid_input_dim = len(variables)  # Input weather variables
        mesh_node_dim = 6  # 3D position + cos(lat) + sin(lon) + cos(lon)
        edge_dim = 4  # 3D relative position + distance

        self.encoder = Grid2MeshEncoder(
            grid_node_dim=grid_input_dim,
            mesh_node_dim=mesh_node_dim,
            edge_dim=edge_dim,
            hidden_size=hidden_size,
            output_size=hidden_size
        )
        self.processor = GNNProcessor(hidden_size, depth, edge_dim=edge_dim)
        self.decoder = Mesh2GridDecoder(
            mesh_dim=hidden_size,
            grid_dim=hidden_size,
            edge_dim=edge_dim,
            hidden_size=hidden_size,
            output_vars=len(variables)
        )

    def _precompute_spatial_features(self):
        """Precompute static spatial features for nodes and edges."""
        # Convert to numpy for feature computation
        grid_lat = self.graph_data.grid_lat.cpu().numpy()
        grid_lon = self.graph_data.grid_lon.cpu().numpy()
        mesh_vertices = self.graph_data.node_coords.cpu().numpy()

        # Grid node features (spatial only, weather data added at runtime)
        grid_node_features = compute_grid_node_features(grid_lat, grid_lon)

        # Mesh node features
        mesh_node_features = compute_mesh_node_features(mesh_vertices)

        # Grid2Mesh edge features
        g2m_edge_features = compute_bipartite_edge_features(
            grid_lat, grid_lon, mesh_vertices,
            self.graph_data.grid2mesh_edge_index.cpu().numpy(),
            grid_to_mesh=True
        )

        # Mesh2Grid edge features
        m2g_edge_features = compute_bipartite_edge_features(
            grid_lat, grid_lon, mesh_vertices,
            self.graph_data.mesh2grid_edge_index.cpu().numpy(),
            grid_to_mesh=False
        )

        # Mesh edge features
        mesh_edge_features = compute_mesh_edge_features(
            mesh_vertices,
            self.graph_data.edge_index.cpu().numpy()
        )

        # Register as buffers
        self.register_buffer('grid_spatial_features', grid_node_features)
        self.register_buffer('mesh_node_features', mesh_node_features)
        self.register_buffer('grid2mesh_edge_features', g2m_edge_features)
        self.register_buffer('mesh2grid_edge_features', m2g_edge_features)
        self.register_buffer('mesh_edge_features', mesh_edge_features)

    def _create_prediction_batch(self, input_batch: WeatherBatch, predictions: torch.Tensor) -> WeatherBatch:
        predictions = predictions.unsqueeze(1)

        if input_batch.forecast_horizon > 1:
            predictions = predictions.repeat(1, input_batch.forecast_horizon, 1, 1, 1)

        B, T, V, H, W = predictions.shape
        channel_idx = 0

        surface_targets = None
        if input_batch.surface_inputs is not None:
            n_surf = input_batch.n_surface_variables
            surface_targets = predictions[:, :, channel_idx:channel_idx + n_surf, :, :]
            channel_idx += n_surf

        atmospheric_targets = None
        if input_batch.atmospheric_inputs is not None:
            n_atmos = input_batch.n_atmospheric_variables
            n_levels = input_batch.n_levels
            flat_channels = n_atmos * n_levels
            atmos_flat = predictions[:, :, channel_idx:channel_idx + flat_channels, :, :]
            atmospheric_targets = atmos_flat.reshape(B, T, n_atmos, n_levels, H, W)

        return WeatherBatch(
            surface_inputs=input_batch.surface_inputs,
            surface_targets=surface_targets,
            atmospheric_inputs=input_batch.atmospheric_inputs,
            atmospheric_targets=atmospheric_targets,
            input_timestamps=input_batch.input_timestamps,
            target_timestamps=input_batch.target_timestamps,
            surface_variable_names=input_batch.surface_variable_names,
            atmospheric_variable_names=input_batch.atmospheric_variable_names,
            pressure_levels=input_batch.pressure_levels,
            spatial_coords=input_batch.spatial_coords,
            sample_indices=input_batch.sample_indices
        )

    def validate_weather_batch(self, weather_batch: WeatherBatch) -> None:
        """Validate WeatherBatch compatibility."""
        if weather_batch.spatial_shape != self.img_size:
            raise ValueError(
                f"Spatial mismatch: expected {self.img_size}, "
                f"got {weather_batch.spatial_shape}"
            )

        if weather_batch.n_variables != len(self.variables):
            raise ValueError(
                f"Variable count mismatch: expected {len(self.variables)}, "
                f"got {weather_batch.n_variables}"
            )

    def forward_graph(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch_info: Optional[Dict] = None
    ) -> Tensor:
        """
        Pure mesh graph forward pass with edge attributes.

        Args:
            x: (B, N, C)
            edge_index: (2, E)
            edge_attr: (E, D_edge) - optional edge features
            batch_info: Optional metadata (not used in GraphCast)

        Returns:
            (B, N, C_out)
        """
        B, N, C = x.shape
        x = x.reshape(B * N, C)

        # Process with edge attributes
        x = self.processor(x, edge_index, edge_attr)
        x = x.reshape(B, N, -1)
        return x

    def get_edge_index(self) -> Tensor:
        return self.mesh_edge_index

    def forward(self, batch: WeatherBatch) -> WeatherBatch:
        """
        Forward pass using full GraphCast architecture:
        Grid → Grid2Mesh GNN → Mesh GNN → Mesh2Grid GNN → Grid
        """
        # Get flattened input grid (B, C, H, W)
        x_grid = batch.flatten_inputs()[:, -1]  # Use last timestep
        B, C, H, W = x_grid.shape

        # Reshape grid to (B, H*W, C) for node-wise processing
        grid_features = x_grid.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Expand mesh node features for batch
        mesh_features = self.mesh_node_features.unsqueeze(0).expand(B, -1, -1)

        # Grid2Mesh encoding (with bipartite GNN)
        latent_mesh, latent_grid = self.encoder(
            grid_features=grid_features.reshape(B * H * W, C),
            mesh_features=mesh_features.reshape(B * self.graph_data.num_nodes, -1),
            edge_index=self.grid2mesh_edge_index,
            edge_attr=self.grid2mesh_edge_features
        )

        # Reshape back to batch format
        latent_mesh = latent_mesh.reshape(B, self.graph_data.num_nodes, -1)
        latent_grid = latent_grid.reshape(B, H * W, -1)

        # Mesh processing (GNN on mesh nodes)
        processed_mesh = self.forward_graph(
            latent_mesh,
            self.mesh_edge_index,
            edge_attr=self.mesh_edge_features
        )

        # Mesh2Grid decoding (with bipartite GNN)
        grid_output = self.decoder(
            mesh_features=processed_mesh.reshape(B * self.graph_data.num_nodes, -1),
            grid_features=latent_grid.reshape(B * H * W, -1),
            edge_index=self.mesh2grid_edge_index,
            edge_attr=self.mesh2grid_edge_features
        )

        # Reshape back to grid format (B, C, H, W)
        predictions = grid_output.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self._create_prediction_batch(batch, predictions)
