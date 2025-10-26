import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch

from ..utils.logging import get_logger

logger = get_logger(__name__)


class NormalizationManager:
    def __init__(self, stats_path: Union[str, Path]):
        self.stats_path = Path(stats_path)
        with open(self.stats_path, "r") as f:
            self.stats = json.load(f)

    def get_variable_stats(self, variable_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        means = [self.stats["means"][var_name] for var_name in variable_names]
        stds = [self.stats["stds"][var_name] for var_name in variable_names]
        return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)

    def normalize_surface(self, data: torch.Tensor, variable_names: List[str]) -> torch.Tensor:
        """
        Normalize surface variables: (data - mean) / std

        Args:
            data: Surface variable tensor of shape (time, n_vars, lat, lon)
            variable_names: List of surface variable names

        Returns:
            Normalized tensor of same shape
        """
        means, stds = self.get_variable_stats(variable_names)
        means_tensor = torch.from_numpy(means).to(data.device).view(1, -1, 1, 1)
        stds_tensor = torch.from_numpy(stds).to(data.device).view(1, -1, 1, 1)
        return (data - means_tensor) / stds_tensor

    def unnormalize_surface(self, data: torch.Tensor, variable_names: List[str]) -> torch.Tensor:
        """
        Reverse normalization for surface variables: data * std + mean

        Args:
            data: Normalized surface variable tensor of shape (time, n_vars, lat, lon)
            variable_names: List of surface variable names

        Returns:
            Unnormalized tensor of same shape
        """
        means, stds = self.get_variable_stats(variable_names)
        means_tensor = torch.from_numpy(means).to(data.device).view(1, -1, 1, 1)
        stds_tensor = torch.from_numpy(stds).to(data.device).view(1, -1, 1, 1)
        return data * stds_tensor + means_tensor

    def normalize_atmospheric(
        self, data: torch.Tensor, variable_names: List[str], pressure_levels: np.ndarray
    ) -> torch.Tensor:
        """
        Normalize atmospheric variables: (data - mean) / std

        Args:
            data: Atmospheric variable tensor of shape (time, n_vars, n_levels, lat, lon)
            variable_names: List of base atmospheric variable names (e.g., ['geopotential', 'temperature'])
            pressure_levels: Array of pressure levels (e.g., [50, 100, 200, 300])

        Returns:
            Normalized tensor of same shape
        """
        # Expand variable names with pressure levels to match normalization_stats.json format
        # e.g., 'geopotential' + [50, 100, 200, 300] -> ['geopotential_50', 'geopotential_100', ...]
        expanded_names = [f"{var}_{level}" for var in variable_names for level in pressure_levels]
        means, stds = self.get_variable_stats(expanded_names)

        # Reshape from flat (n_vars * n_levels,) to (1, n_vars, n_levels, 1, 1) for broadcasting
        n_vars = len(variable_names)
        n_levels = len(pressure_levels)
        means_tensor = torch.from_numpy(means).to(data.device).view(1, n_vars, n_levels, 1, 1)
        stds_tensor = torch.from_numpy(stds).to(data.device).view(1, n_vars, n_levels, 1, 1)
        return (data - means_tensor) / stds_tensor

    def unnormalize_atmospheric(
        self, data: torch.Tensor, variable_names: List[str], pressure_levels: np.ndarray
    ) -> torch.Tensor:
        """
        Reverse normalization for atmospheric variables: data * std + mean

        Args:
            data: Normalized atmospheric variable tensor of shape (time, n_vars, n_levels, lat, lon)
            variable_names: List of base atmospheric variable names (e.g., ['geopotential', 'temperature'])
            pressure_levels: Array of pressure levels (e.g., [50, 100, 200, 300])

        Returns:
            Unnormalized tensor of same shape
        """
        # Expand variable names with pressure levels to match normalization_stats.json format
        expanded_names = [f"{var}_{level}" for var in variable_names for level in pressure_levels]
        means, stds = self.get_variable_stats(expanded_names)

        # Reshape from flat (n_vars * n_levels,) to (1, n_vars, n_levels, 1, 1) for broadcasting
        n_vars = len(variable_names)
        n_levels = len(pressure_levels)
        means_tensor = torch.from_numpy(means).to(data.device).view(1, n_vars, n_levels, 1, 1)
        stds_tensor = torch.from_numpy(stds).to(data.device).view(1, n_vars, n_levels, 1, 1)
        return data * stds_tensor + means_tensor