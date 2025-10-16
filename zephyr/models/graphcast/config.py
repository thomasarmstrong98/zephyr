from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    """Configuration for GraphCast weather prediction model."""
    img_size: Tuple[int, int]
    mesh_levels: int
    hidden_size: int
    depth: int
    mlp_ratio: float = 4.0
