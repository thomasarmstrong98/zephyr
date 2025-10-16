"""Influence analysis for graph-based weather models."""

from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import total_influence

from ..models.base import GraphWeatherModel
from ..data.structures import WeatherBatch


def compute_influence(
    model: GraphWeatherModel,
    data_batch: WeatherBatch,
    max_hops: int = 10,
    num_samples: Optional[int] = None,
    normalize: bool = True,
    device: str = "cuda"
) -> Dict[str, Tensor]:
    """
    Compute Jacobian-based influence for a graph weather model.

    Args:
        model: GraphWeatherModel instance
        data_batch: Input weather batch
        max_hops: Maximum hop distance to analyze
        num_samples: Number of random seed nodes (None = all nodes)
        normalize: Normalize influence by hop 0
        device: Computation device

    Returns:
        Dictionary with:
            - influence: (max_hops+1,) tensor of influence per hop
            - receptive_field: Scalar receptive field breadth R
            - num_nodes: Number of graph nodes
    """
    model.eval()
    model.to(device)
    data_batch = data_batch.to_device(device)

    x = data_batch.flatten_inputs()[:, -1]
    x_graph, num_nodes = model.graph_topology.grid_to_graph(x)
    edge_index = model.get_edge_index().to(device)

    B, N, C = x_graph.shape
    x_graph_flat = x_graph.reshape(B * N, C)

    pyg_data = Data(x=x_graph_flat, edge_index=edge_index)

    def model_wrapper(x, edge_index):
        x = x.reshape(B, N, -1)
        batch_info = {}
        if hasattr(model, 't_embedder'):
            forecast_delta = data_batch.get_forecast_deltas()[:, 0]
            batch_info['forecast_delta'] = forecast_delta
        out = model.forward_graph(x, edge_index, batch_info)
        return out.reshape(B * N, -1)

    with torch.no_grad():
        influence, R = total_influence(
            model_wrapper,
            pyg_data,
            max_hops=max_hops,
            num_samples=num_samples,
            normalize=normalize,
            average=True,
            device=device,
            vectorize=True
        )

    return {
        "influence": influence,
        "receptive_field": R,
        "num_nodes": num_nodes,
        "max_hops": max_hops
    }


def compute_autoregressive_influence(
    model: GraphWeatherModel,
    data_batch: WeatherBatch,
    num_rollout_steps: int,
    max_hops: int = 10,
    num_samples: Optional[int] = None,
    device: str = "cuda"
) -> List[Dict[str, Tensor]]:
    """
    Compute influence evolution across autoregressive rollout steps.

    Args:
        model: GraphWeatherModel instance
        data_batch: Initial weather batch
        num_rollout_steps: Number of autoregressive steps to analyze
        max_hops: Maximum hop distance
        num_samples: Number of random seed nodes
        device: Computation device

    Returns:
        List of influence dictionaries, one per rollout step
    """
    model.eval()
    model.to(device)

    results = []
    current_batch = data_batch.to_device(device)

    for step in range(num_rollout_steps):
        influence_dict = compute_influence(
            model, current_batch, max_hops, num_samples, normalize=True, device=device
        )
        influence_dict['rollout_step'] = step
        results.append(influence_dict)

        with torch.no_grad():
            next_batch = model(current_batch)
            current_batch = WeatherBatch(
                surface_inputs=next_batch.surface_targets,
                surface_targets=None,
                atmospheric_inputs=next_batch.atmospheric_targets,
                atmospheric_targets=None,
                input_timestamps=current_batch.target_timestamps,
                target_timestamps=current_batch.target_timestamps,
                surface_variable_names=current_batch.surface_variable_names,
                atmospheric_variable_names=current_batch.atmospheric_variable_names,
                pressure_levels=current_batch.pressure_levels,
                spatial_coords=current_batch.spatial_coords,
                sample_indices=current_batch.sample_indices
            )

    return results


def compute_per_variable_influence(
    model: GraphWeatherModel,
    data_batch: WeatherBatch,
    variable_names: List[str],
    max_hops: int = 10,
    device: str = "cuda"
) -> Dict[str, Dict[str, Tensor]]:
    """
    Compute influence separately for each weather variable.

    Args:
        model: GraphWeatherModel instance
        data_batch: Input weather batch
        variable_names: List of variable names to analyze
        max_hops: Maximum hop distance
        device: Computation device

    Returns:
        Dictionary mapping variable names to their influence results
    """
    model.eval()
    model.to(device)
    data_batch = data_batch.to_device(device)

    all_vars = data_batch.surface_variable_names + data_batch.atmospheric_variable_names
    results = {}

    for var_name in variable_names:
        if var_name not in all_vars:
            continue

        var_type, var_idx = data_batch.get_variable_index(var_name)

        if var_type == "surface":
            masked_batch = WeatherBatch(
                surface_inputs=data_batch.surface_inputs[:, :, var_idx:var_idx+1],
                surface_targets=data_batch.surface_targets,
                atmospheric_inputs=None,
                atmospheric_targets=None,
                input_timestamps=data_batch.input_timestamps,
                target_timestamps=data_batch.target_timestamps,
                surface_variable_names=[var_name],
                atmospheric_variable_names=[],
                pressure_levels=None,
                spatial_coords=data_batch.spatial_coords,
                sample_indices=data_batch.sample_indices
            )
        else:
            masked_batch = WeatherBatch(
                surface_inputs=None,
                surface_targets=None,
                atmospheric_inputs=data_batch.atmospheric_inputs[:, :, var_idx:var_idx+1],
                atmospheric_targets=data_batch.atmospheric_targets,
                input_timestamps=data_batch.input_timestamps,
                target_timestamps=data_batch.target_timestamps,
                surface_variable_names=[],
                atmospheric_variable_names=[var_name],
                pressure_levels=data_batch.pressure_levels,
                spatial_coords=data_batch.spatial_coords,
                sample_indices=data_batch.sample_indices
            )

        influence_dict = compute_influence(
            model, masked_batch, max_hops, num_samples=100, device=device
        )
        results[var_name] = influence_dict

    return results
