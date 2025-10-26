"""
Icosahedral mesh graph construction for GraphCast-style weather models.

This module implements icosahedral mesh generation and bipartite graph construction
for grid-mesh connectivity, based on the GraphCast architecture (Lam et al., Science 2023).

Reference:
    GraphCast: Learning skillful medium-range global weather forecasting
    Lam et al., Science 382, 1416-1421 (2023)
    https://github.com/google-deepmind/graphcast
"""

from typing import Tuple

import icosphere
import numpy as np
import scipy.spatial
import torch
import trimesh

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


def _grid_lat_lon_to_xyz(grid_latitude: np.ndarray, grid_longitude: np.ndarray) -> np.ndarray:
    """
    Convert lat-lon grid to 3D Cartesian coordinates on unit sphere.

    Args:
        grid_latitude: 1D array of latitudes in degrees
        grid_longitude: 1D array of longitudes in degrees

    Returns:
        Array of shape (H, W, 3) with Cartesian coordinates
    """
    # Create meshgrid: phi (longitude), theta (colatitude)
    phi_grid, theta_grid = np.meshgrid(
        np.deg2rad(grid_longitude),
        np.deg2rad(90 - grid_latitude)
    )

    # Convert to Cartesian coordinates
    x = np.cos(phi_grid) * np.sin(theta_grid)
    y = np.sin(phi_grid) * np.sin(theta_grid)
    z = np.cos(theta_grid)

    return np.stack([x, y, z], axis=-1)


def radius_query_indices(
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    mesh_vertices: np.ndarray,
    radius: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find mesh vertices within a radius of each grid point (for Grid2Mesh).

    Args:
        grid_latitude: 1D array of latitudes in degrees (H,)
        grid_longitude: 1D array of longitudes in degrees (W,)
        mesh_vertices: Mesh vertex positions in 3D (N, 3)
        radius: Query radius for neighbor search

    Returns:
        grid_indices: Grid point indices (E,)
        mesh_indices: Corresponding mesh vertex indices (E,)
    """
    # Convert grid to 3D coordinates
    grid_positions = _grid_lat_lon_to_xyz(grid_latitude, grid_longitude)
    H, W = grid_positions.shape[:2]
    grid_positions_flat = grid_positions.reshape(-1, 3)

    # Build KD-tree for mesh vertices
    kd_tree = scipy.spatial.cKDTree(mesh_vertices)

    # Query for all mesh vertices within radius of each grid point
    query_indices = kd_tree.query_ball_point(x=grid_positions_flat, r=radius)

    # Flatten results into edge list
    grid_edge_indices = []
    mesh_edge_indices = []

    for grid_idx, mesh_neighbors in enumerate(query_indices):
        if len(mesh_neighbors) > 0:
            grid_edge_indices.append(np.repeat(grid_idx, len(mesh_neighbors)))
            mesh_edge_indices.append(mesh_neighbors)

    grid_edge_indices = np.concatenate(grid_edge_indices, axis=0).astype(np.int64)
    mesh_edge_indices = np.concatenate(mesh_edge_indices, axis=0).astype(np.int64)

    return grid_edge_indices, mesh_edge_indices


def in_mesh_triangle_indices(
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find containing triangle vertices for each grid point (for Mesh2Grid).

    Each grid point is connected to the 3 vertices of its containing triangle.

    Args:
        grid_latitude: 1D array of latitudes in degrees (H,)
        grid_longitude: 1D array of longitudes in degrees (W,)
        mesh_vertices: Mesh vertex positions in 3D (N, 3)
        mesh_faces: Mesh triangle faces (F, 3)

    Returns:
        grid_indices: Grid point indices, repeated 3x per point (H*W*3,)
        mesh_indices: Triangle vertex indices (H*W*3,)
    """
    # Convert grid to 3D coordinates
    grid_positions = _grid_lat_lon_to_xyz(grid_latitude, grid_longitude)
    H, W = grid_positions.shape[:2]
    grid_positions_flat = grid_positions.reshape(-1, 3)

    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)

    # Find closest point and containing triangle for each grid point
    closest_points, distances, triangle_ids = mesh.nearest.on_surface(grid_positions_flat)

    # Get the 3 vertices of each containing triangle
    containing_triangles = mesh_faces[triangle_ids]  # (H*W, 3)

    # Create edge list: each grid point connects to 3 triangle vertices
    grid_indices = np.repeat(np.arange(H * W), 3)
    mesh_indices = containing_triangles.flatten()

    return grid_indices.astype(np.int64), mesh_indices.astype(np.int64)


def create_icosahedral_graph(
    levels: int,
    grid_shape: Tuple[int, int],
    radius: float = 0.6
) -> GraphData:
    """
    Create an icosahedral mesh graph with bipartite grid-mesh connectivity.

    This function generates a multi-resolution icosahedral mesh similar to GraphCast
    and computes bipartite graph connections between the mesh nodes and a regular
    lat-lon grid using radius queries and triangle containment.

    Args:
        levels: Number of refinement levels for the icosahedron
        grid_shape: Shape of the lat-lon grid (H, W)
        radius: Query radius for grid2mesh connections (default: 0.6)

    Returns:
        GraphData containing the mesh structure and bipartite connectivity
    """
    # Generate icosahedral mesh
    vertices, faces = icosphere.icosphere(nu=levels)
    edge_index = _build_edges(faces, len(vertices))

    # Create regular lat-lon grid
    H, W = grid_shape
    grid_lat = np.linspace(90, -90, H)
    grid_lon = np.linspace(-180, 180, W, endpoint=False)

    # Compute maximum edge length for radius scaling
    edge_lengths = np.linalg.norm(
        vertices[edge_index[0]] - vertices[edge_index[1]], axis=1
    )
    max_edge_length = edge_lengths.max()
    query_radius = radius * max_edge_length

    # Grid2Mesh: radius query to find mesh nodes near each grid point
    grid_indices_g2m, mesh_indices_g2m = radius_query_indices(
        grid_lat, grid_lon, vertices, query_radius
    )

    # Mesh2Grid: triangle containment for exact interpolation
    grid_indices_m2g, mesh_indices_m2g = in_mesh_triangle_indices(
        grid_lat, grid_lon, vertices, faces
    )

    # Create edge indices in PyG format (2, E)
    grid2mesh_edges = np.stack([grid_indices_g2m, mesh_indices_g2m], axis=0)
    mesh2grid_edges = np.stack([mesh_indices_m2g, grid_indices_m2g], axis=0)

    return GraphData(
        node_coords=torch.from_numpy(vertices).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        num_nodes=len(vertices),
        faces=torch.from_numpy(faces).long(),
        grid_lat=torch.from_numpy(grid_lat).float(),
        grid_lon=torch.from_numpy(grid_lon).float(),
        num_grid_nodes=H * W,
        grid2mesh_edge_index=torch.from_numpy(grid2mesh_edges).long(),
        mesh2grid_edge_index=torch.from_numpy(mesh2grid_edges).long()
    )
