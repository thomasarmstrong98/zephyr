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
    Container for graph structure and bipartite grid-mesh connectivity.

    Attributes:
        # Mesh graph structure
        node_coords: Mesh node coordinates in 3D (N, 3)
        edge_index: Mesh edge connectivity in COO format (2, E_mesh)
        num_nodes: Number of mesh nodes N
        faces: Mesh triangle faces (F, 3)

        # Grid information
        grid_lat: Grid latitudes in degrees (H,)
        grid_lon: Grid longitudes in degrees (W,)
        num_grid_nodes: Total number of grid points H*W

        # Bipartite graph connectivity
        grid2mesh_edge_index: Grid-to-mesh edges (2, E_g2m)
        mesh2grid_edge_index: Mesh-to-grid edges (2, E_m2g)
    """
    node_coords: torch.Tensor
    edge_index: torch.Tensor
    num_nodes: int
    faces: torch.Tensor
    grid_lat: torch.Tensor
    grid_lon: torch.Tensor
    num_grid_nodes: int
    grid2mesh_edge_index: torch.Tensor
    mesh2grid_edge_index: torch.Tensor
