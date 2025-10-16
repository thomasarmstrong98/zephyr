"""
Icosahedral mesh graph construction for GraphCast-style weather models.

This module implements icosahedral mesh generation and grid-graph interpolation
based on the GraphCast architecture (Lam et al., Science 2023).

Reference:
    GraphCast: Learning skillful medium-range global weather forecasting
    Lam et al., Science 382, 1416-1421 (2023)
    https://github.com/google-deepmind/graphcast
"""

from typing import Tuple

import icosphere
import numpy as np
import torch
from torch import Tensor

from .base import GraphData


def _xyz_to_latlon(xyz: np.ndarray) -> np.ndarray:
    """Convert 3D Cartesian coordinates to lat-lon coordinates."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(y, x))
    return np.stack([lat, lon], axis=1)


def _build_edges(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    """Build bidirectional edge list from triangle faces."""
    edges = set()
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            edges.add((min(v1, v2), max(v1, v2)))

    edge_array = np.array(list(edges), dtype=np.int64)
    bidirectional = np.concatenate([edge_array, edge_array[:, ::-1]], axis=0)
    return bidirectional.T


def _compute_interpolation_weights(
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    k: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute k-NN interpolation weights using inverse distance weighting.

    Args:
        source_coords: Source coordinate array (N_source, 2)
        target_coords: Target coordinate array (N_target, 2)
        k: Number of nearest neighbors

    Returns:
        indices: Neighbor indices (N_target, k)
        weights: Interpolation weights (N_target, k)
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(source_coords)
    distances, indices = tree.query(target_coords, k=k)

    eps = 1e-10
    inv_distances = 1.0 / (distances + eps)
    weights = inv_distances / inv_distances.sum(axis=1, keepdims=True)

    return indices, weights


def create_icosahedral_graph(levels: int, grid_shape: Tuple[int, int]) -> GraphData:
    """
    Create an icosahedral mesh graph with grid interpolation weights.

    This function generates a multi-resolution icosahedral mesh similar to GraphCast
    and computes bidirectional interpolation weights between the mesh nodes and
    a regular lat-lon grid.

    Args:
        levels: Number of refinement levels for the icosahedron
        grid_shape: Shape of the lat-lon grid (H, W)

    Returns:
        GraphData containing the mesh structure and interpolation weights
    """
    # Generate icosahedral mesh
    vertices, faces = icosphere.icosphere(nu=levels)
    node_lat_lon = _xyz_to_latlon(vertices)
    edge_index = _build_edges(faces, len(vertices))

    # Create regular lat-lon grid
    H, W = grid_shape
    lats = np.linspace(90, -90, H)
    lons = np.linspace(-180, 180, W, endpoint=False)
    grid_lons, grid_lats = np.meshgrid(lons, lats)
    grid_coords = np.stack([grid_lats.flatten(), grid_lons.flatten()], axis=1)

    # Compute interpolation weights: nodes -> grid
    node_to_grid_idx, node_to_grid_weights = _compute_interpolation_weights(
        grid_coords, node_lat_lon, k=4
    )

    # Compute interpolation weights: grid -> nodes
    grid_to_node_idx, grid_to_node_weights = _compute_interpolation_weights(
        node_lat_lon, grid_coords, k=4
    )
    grid_to_node_idx = grid_to_node_idx.reshape(H, W, -1)
    grid_to_node_weights = grid_to_node_weights.reshape(H, W, -1)

    return GraphData(
        node_coords=torch.from_numpy(node_lat_lon).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        num_nodes=len(vertices),
        grid_to_node_idx=torch.from_numpy(grid_to_node_idx).long(),
        grid_to_node_weights=torch.from_numpy(grid_to_node_weights).float(),
        node_to_grid_idx=torch.from_numpy(node_to_grid_idx).long(),
        node_to_grid_weights=torch.from_numpy(node_to_grid_weights).float()
    )


def grid_to_graph_ico(x_grid: Tensor, graph_data: GraphData) -> Tuple[Tensor, int]:
    """
    Convert grid representation to graph representation using interpolation.

    Args:
        x_grid: Grid tensor of shape (B, C, H, W)
        graph_data: Graph structure with interpolation weights

    Returns:
        x_graph: Graph tensor of shape (B, N, C)
        num_nodes: Number of nodes N
    """
    B, C, H, W = x_grid.shape
    k = graph_data.grid_to_node_idx.shape[-1]

    x_grid = x_grid.permute(0, 2, 3, 1)  # (B, H, W, C)
    indices = graph_data.grid_to_node_idx.to(x_grid.device)
    weights = graph_data.grid_to_node_weights.to(x_grid.device)

    x_flat = x_grid.reshape(B, H * W, C)
    indices_flat = indices.reshape(H * W, k)

    x_neighbors = x_flat[:, indices_flat]  # (B, H*W, k, C)
    weights_expanded = weights.reshape(H * W, k, 1)

    x_nodes = (x_neighbors * weights_expanded).sum(dim=2)  # (B, H*W, C)
    return x_nodes, graph_data.num_nodes


def graph_to_grid_ico(x_nodes: Tensor, graph_data: GraphData, grid_shape: Tuple[int, int]) -> Tensor:
    """
    Convert graph representation to grid representation using interpolation.

    Args:
        x_nodes: Graph tensor of shape (B, N, C)
        graph_data: Graph structure with interpolation weights
        grid_shape: Target grid shape (H, W)

    Returns:
        x_grid: Grid tensor of shape (B, C, H, W)
    """
    B, N, C = x_nodes.shape
    H, W = grid_shape
    k = graph_data.node_to_grid_idx.shape[-1]

    indices = graph_data.node_to_grid_idx.to(x_nodes.device)
    weights = graph_data.node_to_grid_weights.to(x_nodes.device)

    x_grid_flat = torch.zeros(B, H * W, C, device=x_nodes.device)

    for b in range(B):
        for i in range(k):
            grid_idx = indices[:, i]
            w = weights[:, i:i+1]
            x_grid_flat[b].scatter_add_(0, grid_idx.unsqueeze(1).expand(-1, C), x_nodes[b] * w)

    return x_grid_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
