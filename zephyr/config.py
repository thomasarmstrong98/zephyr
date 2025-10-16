from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    zarr_path: str
    train_years: list[int]
    val_years: list[int]
    sequence_length: int = 1
    forecast_horizon: int = 1
    normalize: bool = True

    def __post_init__(self):
        self.zarr_path = Path(self.zarr_path)
