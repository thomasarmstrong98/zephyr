"""
Regular grid graph construction for Stormer-style weather models.

This module implements patch-based grid graph generation with k-NN connectivity
based on the Stormer architecture (Nguyen et al., NeurIPS 2024).

Reference:
    Scaling transformer neural networks for skillful and reliable
    medium-range weather forecasting
    Nguyen et al., NeurIPS 2024
    https://github.com/tung-nd/stormer
"""

from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from .base import GraphData


def create_grid_graph(
    grid_shape: Tuple[int, int],
    patch_size: int = 2,
    k_neighbors: int = 8
) -> GraphData:
    """
    Create a regular grid graph with patch-based nodes and k-NN connectivity.

    This function creates a graph where each node represents a patch of the grid,
    with edges connecting neighboring patches (4-connected or 8-connected).

    Args:
        grid_shape: Shape of the input grid (H, W)
        patch_size: Size of each patch (nodes are patch centers)
        k_neighbors: Number of neighbors (4 or 8)

    Returns:
        GraphData containing the grid graph structure and interpolation weights
    """
    H, W = grid_shape
    h_patches = H // patch_size
    w_patches = W // patch_size
    num_nodes = h_patches * w_patches

    # Compute node coordinates (patch centers)
    node_coords = []
    for i in range(h_patches):
        for j in range(w_patches):
            y = (i + 0.5) * patch_size
            x = (j + 0.5) * patch_size
            node_coords.append([y, x])
    node_coords = np.array(node_coords, dtype=np.float32)

    # Build edge connectivity
    edge_list = []
    for i in range(h_patches):
        for j in range(w_patches):
            node_idx = i * w_patches + j

            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    if k_neighbors == 4 and abs(di) + abs(dj) > 1:
                        continue

                    ni, nj = i + di, j + dj
                    if 0 <= ni < h_patches and 0 <= nj < w_patches:
                        neighbor_idx = ni * w_patches + nj
                        edge_list.append([node_idx, neighbor_idx])

    edge_index = np.array(edge_list, dtype=np.int64).T

    # Grid to node mapping (each grid point maps to one node)
    grid_to_node_idx = np.zeros((H, W, 1), dtype=np.int64)
    grid_to_node_weights = np.ones((H, W, 1), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            patch_i = i // patch_size
            patch_j = j // patch_size
            node_idx = patch_i * w_patches + patch_j
            grid_to_node_idx[i, j, 0] = node_idx

    # Node to grid mapping (each node maps to all pixels in its patch)
    node_to_grid_idx = np.zeros((num_nodes, patch_size * patch_size), dtype=np.int64)
    node_to_grid_weights = np.ones((num_nodes, patch_size * patch_size), dtype=np.float32)
    node_to_grid_weights /= (patch_size * patch_size)  # Average pooling

    for node_idx in range(num_nodes):
        patch_i = node_idx // w_patches
        patch_j = node_idx % w_patches

        local_idx = 0
        for di in range(patch_size):
            for dj in range(patch_size):
                grid_i = patch_i * patch_size + di
                grid_j = patch_j * patch_size + dj
                grid_flat_idx = grid_i * W + grid_j
                node_to_grid_idx[node_idx, local_idx] = grid_flat_idx
                local_idx += 1

    return GraphData(
        node_coords=torch.from_numpy(node_coords).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        num_nodes=num_nodes,
        grid_to_node_idx=torch.from_numpy(grid_to_node_idx).long(),
        grid_to_node_weights=torch.from_numpy(grid_to_node_weights).float(),
        node_to_grid_idx=torch.from_numpy(node_to_grid_idx).long(),
        node_to_grid_weights=torch.from_numpy(node_to_grid_weights).float()
    )


def grid_to_graph_patch(x_grid: Tensor, graph_data: GraphData, patch_size: int) -> Tuple[Tensor, int]:
    """
    Convert grid representation to graph via average pooling over patches.

    Args:
        x_grid: Grid tensor of shape (B, C, H, W)
        graph_data: Graph structure (not used in this implementation)
        patch_size: Size of patches for pooling

    Returns:
        x_graph: Graph tensor of shape (B, N, C)
        num_nodes: Number of nodes N
    """
    B, C, H, W = x_grid.shape
    h_patches = H // patch_size
    w_patches = W // patch_size

    # Reshape and pool
    x_grid = x_grid.reshape(
        B, C, h_patches, patch_size, w_patches, patch_size
    )
    x_nodes = x_grid.permute(0, 2, 4, 1, 3, 5).reshape(
        B, h_patches * w_patches, C, patch_size * patch_size
    )
    x_nodes = x_nodes.mean(dim=-1)

    return x_nodes, graph_data.num_nodes


def graph_to_grid_patch(x_nodes: Tensor, graph_data: GraphData, grid_shape: Tuple[int, int], patch_size: int) -> Tensor:
    """
    Convert graph representation to grid via bilinear unpatchify.

    Args:
        x_nodes: Graph tensor of shape (B, N, C)
        graph_data: Graph structure (not used in this implementation)
        grid_shape: Target grid shape (H, W)
        patch_size: Size of patches

    Returns:
        x_grid: Grid tensor of shape (B, C, H, W)
    """
    B, N, C = x_nodes.shape
    H, W = grid_shape
    h_patches = H // patch_size
    w_patches = W // patch_size

    # Reshape to patch layout
    x_nodes = x_nodes.reshape(B, h_patches, w_patches, C)
    x_nodes = x_nodes.permute(0, 3, 1, 2)  # (B, C, h_patches, w_patches)

    # Bilinear interpolation to full resolution
    x_grid = torch.nn.functional.interpolate(
        x_nodes,
        size=(H, W),
        mode='bilinear',
        align_corners=False
    )

    return x_grid
