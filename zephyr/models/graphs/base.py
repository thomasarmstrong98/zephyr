"""
Graph data structures for weather prediction models.

This module provides simple dataclasses for representing graph structures
used in graph-based weather prediction models.
"""

from dataclasses import dataclass

import torch


@dataclass
class GraphData:
    """
    Container for graph structure and interpolation weights.

    Attributes:
        node_coords: Coordinates of graph nodes (N, D) where D is coordinate dimension
        edge_index: Edge connectivity in COO format (2, E) where E is number of edges
        num_nodes: Number of nodes in the graph
        grid_to_node_idx: Indices for interpolating grid to nodes (H, W, k)
        grid_to_node_weights: Weights for interpolating grid to nodes (H, W, k)
        node_to_grid_idx: Indices for interpolating nodes to grid (N, k)
        node_to_grid_weights: Weights for interpolating nodes to grid (N, k)

    where k is the number of nearest neighbors used for interpolation.
    """
    node_coords: torch.Tensor
    edge_index: torch.Tensor
    num_nodes: int
    grid_to_node_idx: torch.Tensor
    grid_to_node_weights: torch.Tensor
    node_to_grid_idx: torch.Tensor
    node_to_grid_weights: torch.Tensor
