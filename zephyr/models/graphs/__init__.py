"""
Graph construction and interpolation utilities for weather models.

This module provides functional interfaces for creating graph representations
and converting between grid and graph representations.
"""

from .base import GraphData
from .grid import create_grid_graph, graph_to_grid_patch, grid_to_graph_patch
from .icosahedral import create_icosahedral_graph

__all__ = [
    # Data structures
    "GraphData",
    # Icosahedral graph (GraphCast-style)
    "create_icosahedral_graph",
    # Grid graph (Stormer-style)
    "create_grid_graph",
    "grid_to_graph_patch",
    "graph_to_grid_patch",
]
