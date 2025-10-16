from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
from torch import Tensor

from ..data.structures import WeatherBatch


class WeatherModel(ABC):
    """Base interface for weather prediction models."""

    @abstractmethod
    def forward(self, batch: WeatherBatch) -> WeatherBatch:
        """
        Predict future weather states from input batch.

        Args:
            batch: Input weather data with targets

        Returns:
            WeatherBatch with predictions replacing targets
        """
        pass


class GraphWeatherModel(WeatherModel):
    """Extended interface for graph-based weather models."""

    @abstractmethod
    def forward_graph(self, x: Tensor, edge_index: Tensor, batch_info: Optional[Dict] = None) -> Tensor:
        """
        Forward pass using graph representation.

        Args:
            x: Node features (B*N, C) or (B, N, C)
            edge_index: Edge connectivity (2, E)
            batch_info: Optional metadata (forecast_delta, node_count, etc.)

        Returns:
            Node predictions (B*N, C) or (B, N, C)
        """
        pass

    @abstractmethod
    def get_edge_index(self) -> Tensor:
        """Return graph edge connectivity."""
        pass
