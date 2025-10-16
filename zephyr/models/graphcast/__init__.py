import torch
from torch import nn, Tensor
from typing import Dict, List, Optional, Tuple

from .config import Config
from .encoder import GraphEncoder
from .processor import GNNProcessor
from .decoder import GraphDecoder
from ..base import GraphWeatherModel
from ..graphs import create_icosahedral_graph, grid_to_graph_ico, graph_to_grid_ico
from ...data.structures import WeatherBatch


class GraphCast(nn.Module, GraphWeatherModel):
    def __init__(self, img_size: Tuple[int, int], variables: List[str], config: Config):
        super().__init__()
        self.variables = variables
        self.img_size = img_size

        self.graph_data = create_icosahedral_graph(config.mesh_levels, img_size)
        self.register_buffer('edge_index', self.graph_data.edge_index)

        self.encoder = GraphEncoder(len(variables), config.hidden_size)
        self.processor = GNNProcessor(config.hidden_size, config.depth)
        self.decoder = GraphDecoder(config.hidden_size, len(variables))

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

    def get_input_requirements(self) -> dict:
        return {
            'img_size': self.img_size,
            'n_variables': len(self.variables),
            'expected_variables': self.variables
        }

    def validate_weather_batch(self, weather_batch: WeatherBatch) -> None:
        requirements = self.get_input_requirements()

        actual_spatial = weather_batch.spatial_shape
        expected_spatial = requirements['img_size']
        if actual_spatial != expected_spatial:
            raise ValueError(
                f"Spatial dimension mismatch: model expects {expected_spatial}, "
                f"got {actual_spatial}"
            )

        actual_vars = weather_batch.n_variables
        expected_vars = requirements['n_variables']
        if actual_vars != expected_vars:
            raise ValueError(
                f"Variable count mismatch: model expects {expected_vars} (flattened), "
                f"got {actual_vars} ({weather_batch.n_surface_variables} surface + "
                f"{weather_batch.n_atmospheric_variables}x{weather_batch.n_levels} atmospheric)"
            )

        if weather_batch.sequence_length == 0:
            raise ValueError("Empty input sequence not supported")
        if weather_batch.forecast_horizon == 0:
            raise ValueError("Zero forecast horizon not supported")

    def forward_graph(self, x: Tensor, edge_index: Tensor, batch_info: Optional[Dict] = None) -> Tensor:
        """
        Pure graph forward pass.

        Args:
            x: (B, N, C)
            edge_index: (2, E)
            batch_info: Optional metadata

        Returns:
            (B, N, C_out)
        """
        B, N, C = x.shape
        x = x.reshape(B * N, C)
        x = self.processor(x, edge_index)
        x = x.reshape(B, N, -1)
        return x

    def get_edge_index(self) -> Tensor:
        return self.edge_index

    def forward(self, batch: WeatherBatch) -> WeatherBatch:
        x = batch.flatten_inputs()[:, -1]
        x_graph, num_nodes = grid_to_graph_ico(x, self.graph_data)

        x_graph = self.encoder(x_graph)
        x_graph = self.forward_graph(x_graph, self.edge_index, {"num_nodes": num_nodes})
        x_graph = self.decoder(x_graph)

        predictions = graph_to_grid_ico(x_graph, self.graph_data, self.img_size)
        return self._create_prediction_batch(batch, predictions)
