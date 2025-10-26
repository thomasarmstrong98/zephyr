from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn
from torch.nn.init import trunc_normal_

from ...data.structures import WeatherBatch
from ..base import WeatherModel
from .core import Block, OutputLayer, TimestepEmbedder
from .embedding import WeatherEmbedding


@dataclass
class Config:
    """Configuration for Stormer weather prediction model."""

    patch_size: int
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float

    def build(self, variables: List[str], img_size: Tuple[int, int]) -> "Stormer":
        """
        Build a Stormer model from this configuration.

        Args:
            variables: List of variable names the model will predict

        Returns:
            Configured Stormer model instance
        """

        return Stormer(
            img_size=img_size,
            variables=variables,
            patch_size=self.patch_size,
            hidden_size=self.hidden_size,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
        )


class Stormer(nn.Module, WeatherModel):
    def __init__(
        self,
        img_size: Tuple[int, int],
        variables: List[str],
        patch_size: int = 2,
        hidden_size: int = 128,
        depth: int = 24,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        # Store configuration for validation
        self.variables = variables
        self.img_size = img_size

        self.embedding = WeatherEmbedding(
            img_size, len(variables), hidden_size, patch_size, num_heads
        )
        self.embedding_norm_layer = nn.LayerNorm(hidden_size)

        # forecast timdelta encoding
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList(
            [Block(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )

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

    def validate_weather_batch(self, weather_batch: WeatherBatch) -> None:
        """Validate WeatherBatch compatibility."""
        if weather_batch.spatial_shape != self.embedding.img_size:
            raise ValueError(
                f"Spatial mismatch: expected {self.embedding.img_size}, "
                f"got {weather_batch.spatial_shape}"
            )

        if weather_batch.n_variables != self.embedding.n_vars:
            raise ValueError(
                f"Variable count mismatch: expected {self.embedding.n_vars}, "
                f"got {weather_batch.n_variables}"
            )

    def forward(self, weather_batch: WeatherBatch) -> WeatherBatch:
        """
        Forward pass using WeatherBatch structured input.

        Args:
            weather_batch: WeatherBatch containing inputs, targets, and metadata

        Returns:
            WeatherBatch with predicted outputs replacing targets
        """
        # Extract input a - take last timestep for single-step prediction
        x = weather_batch.flatten_inputs()[:, -1]  # Shape: (B, V, H, W)

        # Get forecast timedeltas - take first forecast timestep
        forecast_timedelta = weather_batch.get_forecast_deltas()[:, 0]  # Shape: (B,)

        # Forward pass through model
        print(f"pre-embedding: {x.shape}")
        x = self.embedding(x)
        x = self.embedding_norm_layer(x)
        print(f"post-embedding: {x.shape}")

        forecast_timedelta = self.t_embedder(forecast_timedelta)
        print(f"tdelta: {forecast_timedelta.shape}")
        for block in self.blocks:
            print(f"pre-block: {x.shape}")
            x = block(x, forecast_timedelta)
            print(f"post-block: {x.shape}")

        predictions = self.head(x)  # Shape: (B, V, H, W)

        # Create output WeatherBatch with predictions
        return self._create_prediction_batch(weather_batch, predictions)
