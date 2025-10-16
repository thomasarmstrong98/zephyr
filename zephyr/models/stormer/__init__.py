from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.init import trunc_normal_

from ...data.structures import WeatherBatch
from ..base import GraphWeatherModel
from ..graphs import create_grid_graph, grid_to_graph_patch, graph_to_grid_patch
from .config import Config
from .core import OutputLayer, TimestepEmbedder
from .embedding import WeatherEmbedding
from .graph_layers import GraphProcessor


class Stormer(nn.Module, GraphWeatherModel):
    def __init__(
        self,
        img_size: Tuple[int, int],
        variables: List[str],
        patch_size: int = 2,
        hidden_size: int = 128,
        depth: int = 24,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        k_neighbors: int = 8,
    ):
        super().__init__()

        self.variables = variables
        self.img_size = img_size
        self.patch_size = patch_size

        self.graph_data = create_grid_graph(img_size, patch_size, k_neighbors)
        self.register_buffer("edge_index", self.graph_data.edge_index)

        self.embedding = WeatherEmbedding(
            img_size, len(variables), hidden_size, patch_size, num_heads
        )
        self.embedding_norm_layer = nn.LayerNorm(hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.processor = GraphProcessor(hidden_size, depth, num_heads, mlp_ratio)
        self.head = OutputLayer(hidden_size, patch_size, len(variables), img_size)
        self.init_weights()

    def init_weights(self):

        def _init_weights(m: nn.Module):
            """Initialize transformer layers"""
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

        # Initialize timestep embedding MLP:
        trunc_normal_(self.t_embedder.net.weight, std=0.02)

        # # Zero-out adaLN modulation layers in blocks:
        # for block in self.blocks:
        #     nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        # nn.init.constant_(self.head.linear.weight, 0)
        # nn.init.constant_(self.head.linear.bias, 0)

    def _create_prediction_batch(
        self, input_batch: WeatherBatch, predictions: "torch.Tensor"
    ) -> WeatherBatch:
        """
        Create a new WeatherBatch with predictions replacing targets.

        Args:
            input_batch: Original input WeatherBatch
            predictions: Model predictions of shape (B, V, H, W) where V = flattened channels

        Returns:
            WeatherBatch with predictions as targets
        """
        # Expand predictions to add time dimension: (B, V, H, W) -> (B, 1, V, H, W)
        predictions = predictions.unsqueeze(1)

        # Repeat for all forecast steps if needed
        if input_batch.forecast_horizon > 1:
            predictions = predictions.repeat(1, input_batch.forecast_horizon, 1, 1, 1)

        # Split predictions back into surface and atmospheric components
        B, T, V, H, W = predictions.shape
        channel_idx = 0

        surface_targets = None
        if input_batch.surface_inputs is not None:
            n_surf = input_batch.n_surface_variables
            surface_targets = predictions[:, :, channel_idx : channel_idx + n_surf, :, :]
            channel_idx += n_surf

        atmospheric_targets = None
        if input_batch.atmospheric_inputs is not None:
            n_atmos = input_batch.n_atmospheric_variables
            n_levels = input_batch.n_levels
            flat_channels = n_atmos * n_levels
            atmos_flat = predictions[:, :, channel_idx : channel_idx + flat_channels, :, :]
            # Reshape back to (B, T, n_atmos, n_levels, H, W)
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
            sample_indices=input_batch.sample_indices,
        )

    def get_input_requirements(self) -> dict:
        """
        Get model input requirements for validation.

        Returns:
            Dictionary containing model requirements
        """
        return {
            "img_size": self.embedding.img_size,
            "n_variables": self.embedding.n_vars,
            "patch_size": self.embedding.patch_size,
            "expected_variables": getattr(self, "variables", None),
        }

    def validate_weather_batch(self, weather_batch: WeatherBatch) -> None:
        """
        Validate that WeatherBatch is compatible with this model.

        Args:
            weather_batch: WeatherBatch to validate

        Raises:
            ValueError: If batch is incompatible with model requirements
        """
        requirements = self.get_input_requirements()

        # Check spatial dimensions
        actual_spatial = weather_batch.spatial_shape
        expected_spatial = requirements["img_size"]
        if actual_spatial != expected_spatial:
            raise ValueError(
                f"Spatial dimension mismatch: model expects {expected_spatial}, "
                f"got {actual_spatial}"
            )

        # Check total variable count (flattened)
        actual_vars = weather_batch.n_variables
        expected_vars = requirements["n_variables"]
        if actual_vars != expected_vars:
            raise ValueError(
                f"Variable count mismatch: model expects {expected_vars} (flattened), "
                f"got {actual_vars} ({weather_batch.n_surface_variables} surface + "
                f"{weather_batch.n_atmospheric_variables}x{weather_batch.n_levels} atmospheric)"
            )

        # Check for empty sequences
        if weather_batch.sequence_length == 0:
            raise ValueError("Empty input sequence not supported")
        if weather_batch.forecast_horizon == 0:
            raise ValueError("Zero forecast horizon not supported")

    def forward_graph(
        self, x: Tensor, edge_index: Tensor, batch_info: Optional[Dict] = None
    ) -> Tensor:
        """
        Pure graph forward pass.

        Args:
            x: (B, N, C)
            edge_index: (2, E)
            batch_info: Must contain 'forecast_delta' key

        Returns:
            (B, N, C_out)
        """
        B, N, C = x.shape
        forecast_delta = batch_info.get("forecast_delta")

        x = x.reshape(B * N, C)
        forecast_delta_expanded = (
            forecast_delta.unsqueeze(1).expand(B, N).reshape(B * N, -1).squeeze(-1)
        )
        forecast_delta_emb = self.t_embedder(forecast_delta_expanded)

        x = self.processor(x, edge_index, forecast_delta_emb)
        x = x.reshape(B, N, -1)
        return x

    def get_edge_index(self) -> Tensor:
        return self.edge_index

    def forward(self, weather_batch: WeatherBatch) -> WeatherBatch:
        x = weather_batch.flatten_inputs()[:, -1]
        forecast_timedelta = weather_batch.get_forecast_deltas()[:, 0]

        x = self.embedding(x)
        x = self.embedding_norm_layer(x)

        x_graph, num_nodes = grid_to_graph_patch(x.permute(0, 2, 3, 1), self.graph_data, self.patch_size)
        x_graph = self.forward_graph(
            x_graph, self.edge_index, {"forecast_delta": forecast_timedelta}
        )

        x_out = graph_to_grid_patch(x_graph, self.graph_data, self.img_size, self.patch_size)
        x_out = x_out.permute(0, 3, 1, 2)

        predictions = self.head(
            x_out.permute(0, 2, 3, 1).reshape(x_out.shape[0], -1, x_out.shape[1])
        )

        return self._create_prediction_batch(weather_batch, predictions)
