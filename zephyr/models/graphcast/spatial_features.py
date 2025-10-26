"""
Spatial feature computation for GraphCast bipartite graphs.

This module computes geometric features for grid-mesh and mesh-mesh edges
based on the GraphCast architecture.
"""

import numpy as np
import torch
from torch import Tensor
from typing import Tuple


def _lat_lon_to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Convert latitude/longitude to 3D Cartesian coordinates on unit sphere.

    Args:
        lat: Latitudes in degrees
        lon: Longitudes in degrees

    Returns:
        3D Cartesian coordinates (N, 3)
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return np.stack([x, y, z], axis=-1)


def compute_grid_node_features(
    grid_lat: np.ndarray,
    grid_lon: np.ndarray
) -> Tensor:
    """
    Compute node features for grid points.

    Args:
        grid_lat: Grid latitudes in degrees (H,)
        grid_lon: Grid longitudes in degrees (W,)

    Returns:
        Grid node features (H*W, F) where F is feature dimension
    """
    H, W = len(grid_lat), len(grid_lon)
    grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)
    grid_lat_flat = grid_lat_2d.flatten()
    grid_lon_flat = grid_lon_2d.flatten()

    # Convert to 3D positions
    positions = _lat_lon_to_xyz(grid_lat_flat, grid_lon_flat)

    # Additional features: cos(lat), sin(lon), cos(lon)
    lat_rad = np.deg2rad(grid_lat_flat)
    lon_rad = np.deg2rad(grid_lon_flat)

    features = np.concatenate([
        positions,  # (N, 3) - x, y, z positions
        np.cos(lat_rad)[:, None],  # (N, 1) - latitude feature
        np.sin(lon_rad)[:, None],  # (N, 1) - longitude sine
        np.cos(lon_rad)[:, None],  # (N, 1) - longitude cosine
    ], axis=-1)

    return torch.from_numpy(features).float()


def compute_mesh_node_features(mesh_vertices: np.ndarray) -> Tensor:
    """
    Compute node features for mesh vertices.

    Args:
        mesh_vertices: Mesh vertex positions in 3D (N, 3)

    Returns:
        Mesh node features (N, F)
    """
    # Convert 3D positions to lat/lon
    x, y, z = mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2]
    lat = np.arcsin(z)
    lon = np.arctan2(y, x)

    features = np.concatenate([
        mesh_vertices,  # (N, 3) - x, y, z positions
        np.cos(lat)[:, None],  # (N, 1) - latitude feature
        np.sin(lon)[:, None],  # (N, 1) - longitude sine
        np.cos(lon)[:, None],  # (N, 1) - longitude cosine
    ], axis=-1)

    return torch.from_numpy(features).float()


def compute_edge_features(
    sender_positions: Tensor,
    receiver_positions: Tensor
) -> Tensor:
    """
    Compute edge features for bipartite or mesh edges.

    Args:
        sender_positions: 3D positions of sender nodes (E, 3)
        receiver_positions: 3D positions of receiver nodes (E, 3)

    Returns:
        Edge features (E, F) where F includes:
        - Relative position vector (3D)
        - Edge length (1D)
    """
    # Relative position from sender to receiver
    relative_pos = receiver_positions - sender_positions  # (E, 3)

    # Edge length (Euclidean distance)
    edge_length = torch.norm(relative_pos, dim=-1, keepdim=True)  # (E, 1)

    features = torch.cat([
        relative_pos,  # (E, 3)
        edge_length,   # (E, 1)
    ], dim=-1)

    return features


def compute_bipartite_edge_features(
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    mesh_vertices: np.ndarray,
    edge_index: np.ndarray,
    grid_to_mesh: bool = True
) -> Tensor:
    """
    Compute edge features for bipartite graph (grid ↔ mesh).

    Args:
        grid_lat: Grid latitudes in degrees (H,)
        grid_lon: Grid longitudes in degrees (W,)
        mesh_vertices: Mesh vertex positions in 3D (N, 3)
        edge_index: Edge indices (2, E)
        grid_to_mesh: If True, edges go grid→mesh; if False, mesh→grid

    Returns:
        Edge features (E, F)
    """
    # Get grid positions
    H, W = len(grid_lat), len(grid_lon)
    grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)
    grid_lat_flat = grid_lat_2d.flatten()
    grid_lon_flat = grid_lon_2d.flatten()
    grid_positions = _lat_lon_to_xyz(grid_lat_flat, grid_lon_flat)

    # Get sender and receiver positions
    if grid_to_mesh:
        # edge_index[0] = grid, edge_index[1] = mesh
        sender_pos = grid_positions[edge_index[0]]
        receiver_pos = mesh_vertices[edge_index[1]]
    else:
        # edge_index[0] = mesh, edge_index[1] = grid
        sender_pos = mesh_vertices[edge_index[0]]
        receiver_pos = grid_positions[edge_index[1]]

    sender_pos = torch.from_numpy(sender_pos).float()
    receiver_pos = torch.from_numpy(receiver_pos).float()

    return compute_edge_features(sender_pos, receiver_pos)


def compute_mesh_edge_features(
    mesh_vertices: np.ndarray,
    edge_index: np.ndarray
) -> Tensor:
    """
    Compute edge features for mesh graph.

    Args:
        mesh_vertices: Mesh vertex positions in 3D (N, 3)
        edge_index: Edge indices (2, E)

    Returns:
        Edge features (E, F)
    """
    sender_pos = torch.from_numpy(mesh_vertices[edge_index[0]]).float()
    receiver_pos = torch.from_numpy(mesh_vertices[edge_index[1]]).float()

    return compute_edge_features(sender_pos, receiver_pos)
