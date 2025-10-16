"""
Data structures for atmospheric ML with explicit surface/pressure-level separation.
Preserves vertical atmospheric structure for physical consistency.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch


@dataclass
class WeatherSample:
    """Sample with explicit surface/atmospheric separation for vertical structure preservation."""

    surface_inputs: Optional[torch.Tensor]
    surface_targets: Optional[torch.Tensor]
    atmospheric_inputs: Optional[torch.Tensor]
    atmospheric_targets: Optional[torch.Tensor]
    input_timestamps: pd.DatetimeIndex
    target_timestamps: pd.DatetimeIndex
    surface_variable_names: List[str]
    atmospheric_variable_names: List[str]
    pressure_levels: Optional[np.ndarray]
    spatial_coords: Tuple[np.ndarray, np.ndarray]
    sample_index: Optional[int] = None

    @property
    def _ref(self):
        return self.surface_inputs if self.surface_inputs is not None else self.atmospheric_inputs

    def __post_init__(self):
        assert self.surface_inputs is not None or self.atmospheric_inputs is not None

        seq_len, spatial = self._ref.shape[0], self._ref.shape[-2:]

        if self.surface_inputs is not None:
            assert self.surface_targets is not None
            assert (
                self.surface_inputs.shape[1]
                == len(self.surface_variable_names)
                == self.surface_targets.shape[1]
            )

        if self.atmospheric_inputs is not None:
            assert self.atmospheric_targets is not None and self.pressure_levels is not None
            assert self.atmospheric_inputs.shape[1] == len(self.atmospheric_variable_names)
            assert self.atmospheric_inputs.shape[2] == len(self.pressure_levels)
            assert self.atmospheric_inputs.shape[1:3] == self.atmospheric_targets.shape[1:3]

        assert len(self.input_timestamps) == seq_len
        lats, lons = self.spatial_coords
        assert len(lats) == spatial[0] and len(lons) == spatial[1]

    @property
    def sequence_length(self) -> int:
        return self._ref.shape[0]

    @property
    def forecast_horizon(self) -> int:
        return (
            self.surface_targets if self.surface_targets is not None else self.atmospheric_targets
        ).shape[0]

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        return self._ref.shape[-2:]

    def get_variable_slice(self, var_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if var_name in self.surface_variable_names:
            idx = self.surface_variable_names.index(var_name)
            return self.surface_inputs[:, idx], self.surface_targets[:, idx]
        idx = self.atmospheric_variable_names.index(var_name)
        return self.atmospheric_inputs[:, idx], self.atmospheric_targets[:, idx]

    def forecast_deltas(self) -> torch.Tensor:
        dt = [
            (t - self.input_timestamps[-1]).total_seconds() / 3600.0 for t in self.target_timestamps
        ]
        return torch.tensor(dt, dtype=torch.float32)


@dataclass
class WeatherBatch:
    """Batched atmospheric data with surface/pressure-level separation."""

    surface_inputs: Optional[torch.Tensor]
    surface_targets: Optional[torch.Tensor]
    atmospheric_inputs: Optional[torch.Tensor]
    atmospheric_targets: Optional[torch.Tensor]
    input_timestamps: List[pd.DatetimeIndex]
    target_timestamps: List[pd.DatetimeIndex]
    surface_variable_names: List[str]
    atmospheric_variable_names: List[str]
    pressure_levels: Optional[np.ndarray]
    spatial_coords: Tuple[np.ndarray, np.ndarray]
    sample_indices: Optional[List[int]] = None

    @property
    def _ref(self):
        return self.surface_inputs if self.surface_inputs is not None else self.atmospheric_inputs

    def __post_init__(self):
        assert self.surface_inputs is not None or self.atmospheric_inputs is not None
        B, T = self._ref.shape[:2]

        if self.surface_inputs is not None:
            assert self.surface_targets is not None
            assert self.surface_inputs.shape[2] == len(self.surface_variable_names)

        if self.atmospheric_inputs is not None:
            assert self.atmospheric_targets is not None and self.pressure_levels is not None
            assert self.atmospheric_inputs.shape[2] == len(self.atmospheric_variable_names)
            assert self.atmospheric_inputs.shape[3] == len(self.pressure_levels)

        assert len(self.input_timestamps) == len(self.target_timestamps) == B

    @property
    def batch_size(self) -> int:
        return self._ref.shape[0]

    @property
    def sequence_length(self) -> int:
        return self._ref.shape[1]

    @property
    def forecast_horizon(self) -> int:
        return (
            self.surface_targets if self.surface_targets is not None else self.atmospheric_targets
        ).shape[1]

    @property
    def n_surface_variables(self) -> int:
        return len(self.surface_variable_names)

    @property
    def n_atmospheric_variables(self) -> int:
        return len(self.atmospheric_variable_names)

    @property
    def n_levels(self) -> int:
        return len(self.pressure_levels) if self.pressure_levels is not None else 0

    @property
    def n_variables(self) -> int:
        return self.n_surface_variables + (self.n_atmospheric_variables * self.n_levels)

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        if self.surface_inputs is not None:
            return self.surface_inputs.shape[3], self.surface_inputs.shape[4]
        return self.atmospheric_inputs.shape[4], self.atmospheric_inputs.shape[5]

    def flatten_inputs(self) -> torch.Tensor:
        """Flatten surface and atmospheric inputs into single tensor (B, T, C, H, W)."""
        tensors = []
        if self.surface_inputs is not None:
            tensors.append(self.surface_inputs)
        if self.atmospheric_inputs is not None:
            B, T, V, L, H, W = self.atmospheric_inputs.shape
            tensors.append(self.atmospheric_inputs.reshape(B, T, V * L, H, W))
        return torch.cat(tensors, dim=2) if len(tensors) > 1 else tensors[0]

    def flatten_targets(self) -> torch.Tensor:
        """Flatten surface and atmospheric targets into single tensor (B, T, C, H, W)."""
        tensors = []
        if self.surface_targets is not None:
            tensors.append(self.surface_targets)
        if self.atmospheric_targets is not None:
            B, T, V, L, H, W = self.atmospheric_targets.shape
            tensors.append(self.atmospheric_targets.reshape(B, T, V * L, H, W))
        return torch.cat(tensors, dim=2) if len(tensors) > 1 else tensors[0]

    def to_device(self, device: Union[str, torch.device]) -> "WeatherBatch":
        return WeatherBatch(
            surface_inputs=(
                self.surface_inputs.to(device) if self.surface_inputs is not None else None
            ),
            surface_targets=(
                self.surface_targets.to(device) if self.surface_targets is not None else None
            ),
            atmospheric_inputs=(
                self.atmospheric_inputs.to(device) if self.atmospheric_inputs is not None else None
            ),
            atmospheric_targets=(
                self.atmospheric_targets.to(device)
                if self.atmospheric_targets is not None
                else None
            ),
            input_timestamps=self.input_timestamps,
            target_timestamps=self.target_timestamps,
            surface_variable_names=self.surface_variable_names,
            atmospheric_variable_names=self.atmospheric_variable_names,
            pressure_levels=self.pressure_levels,
            spatial_coords=self.spatial_coords,
            sample_indices=self.sample_indices,
        )

    def get_variable_index(self, var_name: str) -> Tuple[str, int]:
        if var_name in self.surface_variable_names:
            return "surface", self.surface_variable_names.index(var_name)
        if var_name in self.atmospheric_variable_names:
            return "atmospheric", self.atmospheric_variable_names.index(var_name)
        raise ValueError(
            f"Variable '{var_name}' not found in {self.surface_variable_names + self.atmospheric_variable_names}"
        )

    def get_variable_slice(self, var_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        var_type, var_idx = self.get_variable_index(var_name)
        if var_type == "surface":
            return self.surface_inputs[:, :, var_idx], self.surface_targets[:, :, var_idx]
        return self.atmospheric_inputs[:, :, var_idx], self.atmospheric_targets[:, :, var_idx]

    def get_forecast_deltas(self) -> torch.Tensor:
        batch_deltas = []
        for i in range(self.batch_size):
            if not self.input_timestamps[i] or not self.target_timestamps[i]:
                deltas = torch.zeros(self.forecast_horizon)
            else:
                last_input = self.input_timestamps[i][-1]
                deltas = torch.tensor(
                    [(t - last_input).total_seconds() / 3600.0 for t in self.target_timestamps[i]],
                    dtype=torch.float32,
                )
            batch_deltas.append(deltas)
        return torch.stack(batch_deltas)

    def __getitem__(self, idx: int) -> WeatherSample:
        if idx >= self.batch_size:
            raise IndexError(f"Batch index {idx} out of range for batch size {self.batch_size}")

        return WeatherSample(
            surface_inputs=self.surface_inputs[idx] if self.surface_inputs is not None else None,
            surface_targets=self.surface_targets[idx] if self.surface_targets is not None else None,
            atmospheric_inputs=(
                self.atmospheric_inputs[idx] if self.atmospheric_inputs is not None else None
            ),
            atmospheric_targets=(
                self.atmospheric_targets[idx] if self.atmospheric_targets is not None else None
            ),
            input_timestamps=self.input_timestamps[idx],
            target_timestamps=self.target_timestamps[idx],
            surface_variable_names=self.surface_variable_names,
            atmospheric_variable_names=self.atmospheric_variable_names,
            pressure_levels=self.pressure_levels,
            spatial_coords=self.spatial_coords,
            sample_index=self.sample_indices[idx] if self.sample_indices else None,
        )
