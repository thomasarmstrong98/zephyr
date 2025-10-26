from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Config:
    """Configuration for GraphCast weather prediction model."""
    img_size: Tuple[int, int]
    mesh_levels: int
    hidden_size: int
    depth: int
    mlp_ratio: float = 4.0

    def build(self, variables: List[str]) -> "GraphCast":
        """
        Build a GraphCast model from this configuration.

        Args:
            variables: List of variable names the model will predict

        Returns:
            Configured GraphCast model instance
        """
        from . import GraphCast
        return GraphCast(
            img_size=self.img_size,
            variables=variables,
            mesh_levels=self.mesh_levels,
            hidden_size=self.hidden_size,
            depth=self.depth,
            mlp_ratio=self.mlp_ratio
        )
