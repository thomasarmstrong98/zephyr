from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from ..utils.logging import get_logger
from .normalization import NormalizationManager
from .structures import WeatherBatch, WeatherSample

logger = get_logger(__name__)


def collate_fn(batch: List[WeatherSample]) -> WeatherBatch:
    if not batch:
        raise ValueError("Empty batch")

    first = batch[0]

    for i, s in enumerate(batch[1:], 1):
        if s.surface_variable_names != first.surface_variable_names:
            raise ValueError(f"Surface variable mismatch in item {i}")
        if s.atmospheric_variable_names != first.atmospheric_variable_names:
            raise ValueError(f"Atmospheric variable mismatch in item {i}")
        if not np.array_equal(s.spatial_coords[0], first.spatial_coords[0]) or not np.array_equal(
            s.spatial_coords[1], first.spatial_coords[1]
        ):
            raise ValueError(f"Spatial coords mismatch in item {i}")
        if s.pressure_levels is not None and first.pressure_levels is not None:
            if not np.array_equal(s.pressure_levels, first.pressure_levels):
                raise ValueError(f"Pressure levels mismatch in item {i}")

    surface_inputs = (
        torch.stack([s.surface_inputs for s in batch]) if first.surface_inputs is not None else None
    )
    surface_targets = (
        torch.stack([s.surface_targets for s in batch])
        if first.surface_targets is not None
        else None
    )
    atmospheric_inputs = (
        torch.stack([s.atmospheric_inputs for s in batch])
        if first.atmospheric_inputs is not None
        else None
    )
    atmospheric_targets = (
        torch.stack([s.atmospheric_targets for s in batch])
        if first.atmospheric_targets is not None
        else None
    )

    return WeatherBatch(
        surface_inputs=surface_inputs,
        surface_targets=surface_targets,
        atmospheric_inputs=atmospheric_inputs,
        atmospheric_targets=atmospheric_targets,
        input_timestamps=[s.input_timestamps for s in batch],
        target_timestamps=[s.target_timestamps for s in batch],
        surface_variable_names=first.surface_variable_names,
        atmospheric_variable_names=first.atmospheric_variable_names,
        pressure_levels=first.pressure_levels,
        spatial_coords=first.spatial_coords,
        sample_indices=[s.sample_index for s in batch if s.sample_index is not None] or None,
    )


class WeatherBenchDataset(Dataset):
    def __init__(
        self,
        zarr_path: Union[str, Path],
        years: List[int],
        sequence_length: int = 1,
        forecast_horizon: int = 1,
        normalize: bool = True,
        normalization_stats_path: Optional[Union[str, Path]] = None,
    ):
        self.zarr_path = Path(zarr_path)
        self.years = sorted(years)
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.normalize = normalize

        self._load_zarr_store()
        self._setup_normalization(normalization_stats_path)
        self._build_time_indices()

        logger.info(f"Loaded {len(self.time_indices)} samples from {self.years[0]}-{self.years[-1]} "
                   f"({len(self.variable_names)} variables)")

    def _load_zarr_store(self):
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Zarr store not found: {self.zarr_path}")

        self.ds = xr.open_zarr(str(self.zarr_path), chunks="auto", consolidated=False)
        self.time_coordinates = pd.DatetimeIndex(self.ds.time.values)
        self.lats = self.ds.latitude.values
        self.lons = self.ds.longitude.values
        self.variable_names = sorted(list(self.ds.data_vars.keys()))

        self.pressure_level_vars = [v for v in self.variable_names if "level" in self.ds[v].dims]
        self.surface_vars = [v for v in self.variable_names if "level" not in self.ds[v].dims]
        self.total_timesteps = len(self.time_coordinates)
        self.n_lat, self.n_lon = len(self.lats), len(self.lons)

        if "level" in self.ds.coords:
            self.pressure_levels = self.ds.level.values
            self.n_levels = len(self.pressure_levels)
        else:
            self.pressure_levels = None
            self.n_levels = 0

    def _setup_normalization(self, normalization_stats_path: Optional[Union[str, Path]]) -> None:
        if not self.normalize:
            self.normalizer = None
            return
        self.normalizer = NormalizationManager(
            normalization_stats_path or "normalization_stats.json"
        )

    def _build_time_indices(self):
        year_starts = {y: pd.Timestamp(f"{y}-01-01") for y in self.years}
        year_ends = {y: pd.Timestamp(f"{y+1}-01-01") for y in self.years}
        self.time_indices = []
        min_length = self.sequence_length + self.forecast_horizon

        for year in self.years:
            year_mask = (self.time_coordinates >= year_starts[year]) & (
                self.time_coordinates < year_ends[year]
            )
            year_indices = np.where(year_mask)[0]
            if len(year_indices) >= min_length:
                valid_starts = year_indices[: -min_length + 1]
                self.time_indices.extend(valid_starts)

        if not self.time_indices:
            raise ValueError(f"No valid samples for years {self.years}")

    def __len__(self) -> int:
        return len(self.time_indices)

    def __getitem__(self, idx: int) -> WeatherSample:
        start = self.time_indices[idx]
        input_end = start + self.sequence_length
        target_start = input_end
        target_end = target_start + self.forecast_horizon

        surface_inputs, atmospheric_inputs = self._extract_variables(
            self.ds.isel(time=slice(start, input_end))
        )
        surface_targets, atmospheric_targets = self._extract_variables(
            self.ds.isel(time=slice(target_start, target_end))
        )

        if self.normalizer:
            if surface_inputs is not None:
                surface_inputs = self.normalizer.normalize_surface(
                    surface_inputs, self.surface_vars
                )
                surface_targets = self.normalizer.normalize_surface(
                    surface_targets, self.surface_vars
                )
            if atmospheric_inputs is not None:
                atmospheric_inputs = self.normalizer.normalize_atmospheric(
                    atmospheric_inputs, self.pressure_level_vars
                )
                atmospheric_targets = self.normalizer.normalize_atmospheric(
                    atmospheric_targets, self.pressure_level_vars
                )

        return WeatherSample(
            surface_inputs=surface_inputs,
            surface_targets=surface_targets,
            atmospheric_inputs=atmospheric_inputs,
            atmospheric_targets=atmospheric_targets,
            input_timestamps=pd.DatetimeIndex(self.time_coordinates[start:input_end]),
            target_timestamps=pd.DatetimeIndex(self.time_coordinates[target_start:target_end]),
            surface_variable_names=self.surface_vars.copy(),
            atmospheric_variable_names=self.pressure_level_vars.copy(),
            pressure_levels=(
                self.pressure_levels.copy() if self.pressure_levels is not None else None
            ),
            spatial_coords=(np.array(self.lats), np.array(self.lons)),
            sample_index=idx,
        )

    def _extract_variables(
        self, ds: xr.Dataset
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        surface_arrays = [ds[v].values for v in self.surface_vars]
        atmospheric_arrays = [ds[v].values for v in self.pressure_level_vars]

        surface_tensor = (
            torch.from_numpy(np.stack(surface_arrays, axis=1)).float() if surface_arrays else None
        )
        atmospheric_tensor = (
            torch.from_numpy(np.stack(atmospheric_arrays, axis=1)).float()
            if atmospheric_arrays
            else None
        )

        return surface_tensor, atmospheric_tensor

    def unnormalize_sample(self, sample: WeatherSample) -> WeatherSample:
        if not self.normalizer:
            return sample

        si = (
            self.normalizer.unnormalize_surface(
                sample.surface_inputs, sample.surface_variable_names
            )
            if sample.surface_inputs is not None
            else None
        )
        st = (
            self.normalizer.unnormalize_surface(
                sample.surface_targets, sample.surface_variable_names
            )
            if sample.surface_targets is not None
            else None
        )
        ai = (
            self.normalizer.unnormalize_atmospheric(
                sample.atmospheric_inputs, sample.atmospheric_variable_names
            )
            if sample.atmospheric_inputs is not None
            else None
        )
        at = (
            self.normalizer.unnormalize_atmospheric(
                sample.atmospheric_targets, sample.atmospheric_variable_names
            )
            if sample.atmospheric_targets is not None
            else None
        )

        return WeatherSample(
            surface_inputs=si,
            surface_targets=st,
            atmospheric_inputs=ai,
            atmospheric_targets=at,
            input_timestamps=sample.input_timestamps,
            target_timestamps=sample.target_timestamps,
            surface_variable_names=sample.surface_variable_names,
            atmospheric_variable_names=sample.atmospheric_variable_names,
            pressure_levels=sample.pressure_levels,
            spatial_coords=sample.spatial_coords,
            sample_index=sample.sample_index,
        )

    def get_variable_names(self) -> dict:
        return {"surface": self.surface_vars.copy(), "atmospheric": self.pressure_level_vars.copy()}

    def get_pressure_levels(self) -> Optional[np.ndarray]:
        return self.pressure_levels.copy() if self.pressure_levels is not None else None

    def get_spatial_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.lats.copy(), self.lons.copy()

    def get_time_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        return self.time_coordinates[0], self.time_coordinates[-1]

    def info(self) -> dict:
        info = {
            "zarr_path": str(self.zarr_path),
            "years": self.years,
            "n_samples": len(self),
            "sequence_length": self.sequence_length,
            "forecast_horizon": self.forecast_horizon,
            "variables": self.variable_names,
            "surface_variables": self.surface_vars,
            "pressure_level_variables": self.pressure_level_vars,
            "spatial_shape": (self.n_lat, self.n_lon),
            "time_range": self.get_time_range(),
            "normalize": self.normalize,
        }
        if self.pressure_levels is not None:
            info["pressure_levels"] = list(self.pressure_levels)
        return info
