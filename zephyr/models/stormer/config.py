from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    """Configuration for Stormer weather prediction model."""
    img_size: Tuple[int, int]
    patch_size: int
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float