"""Metrics for weather forecasting evaluation."""
import torch
from torchmetrics import Metric, MeanSquaredError, MeanAbsoluteError


class LatitudeWeightedMSE(Metric):
    """MSE with cosine latitude weighting."""

    def __init__(self, nlat: int = 121):
        super().__init__()

        # Cosine latitude weights
        lats = torch.linspace(-90, 90, nlat)
        weights = torch.cos(torch.deg2rad(lats))
        weights = weights / weights.mean()

        # (1, 1, nlat, 1) for broadcasting over (B, V, lat, lon)
        self.register_buffer('lat_weights', weights.view(1, 1, -1, 1))

        self.add_state('sum_squared_error', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update state with predictions and targets.

        Args:
            preds: (B, V, lat, lon)
            target: (B, V, lat, lon)
        """
        squared_error = (preds - target) ** 2
        weighted_error = squared_error * self.lat_weights

        self.sum_squared_error += weighted_error.sum()
        self.count += preds.numel()

    def compute(self):
        """Compute latitude-weighted MSE."""
        return self.sum_squared_error / self.count


class PerVariableMetric(Metric):
    """Compute a metric per variable."""

    def __init__(self, num_vars: int, metric_fn):
        super().__init__()
        self.num_vars = num_vars
        self.metric_fn = metric_fn

        self.add_state('sum_errors', default=torch.zeros(num_vars), dist_reduce_fx='sum')
        self.add_state('counts', default=torch.zeros(num_vars), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update per-variable metrics.

        Args:
            preds: (B, V, lat, lon)
            target: (B, V, lat, lon)
        """
        errors = self.metric_fn(preds, target)  # (B, V, lat, lon)

        # Sum over batch and spatial dims
        var_errors = errors.sum(dim=(0, 2, 3))  # (V,)
        var_counts = torch.tensor(
            preds.shape[0] * preds.shape[2] * preds.shape[3],
            device=preds.device
        ).expand(self.num_vars)

        self.sum_errors += var_errors
        self.counts += var_counts

    def compute(self):
        """Compute per-variable metric."""
        return self.sum_errors / self.counts


def squared_error(preds, target):
    return (preds - target) ** 2


def absolute_error(preds, target):
    return torch.abs(preds - target)


class WeatherMetrics:
    """Metrics collection for weather forecasting.

    Computes RMSE and MAE per variable and overall.
    Automatically excludes forced variables (e.g., land_sea_mask).
    """

    def __init__(
        self,
        variable_names: list[str],
        forced_variables: list[str],
        device: str = 'cuda'
    ):
        # Filter out forced variables
        self.all_variables = variable_names
        self.forced_variables = forced_variables
        self.predictive_variables = [
            v for v in variable_names if v not in forced_variables
        ]
        self.predictive_indices = [
            i for i, v in enumerate(variable_names) if v not in forced_variables
        ]

        self.num_vars = len(self.predictive_variables)
        self.device = device

        # Per-variable metrics
        self.rmse = PerVariableMetric(self.num_vars, squared_error).to(device)
        self.mae = PerVariableMetric(self.num_vars, absolute_error).to(device)

        # Overall metrics
        self.overall_mse = MeanSquaredError().to(device)
        self.overall_mae = MeanAbsoluteError().to(device)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update all metrics.

        Args:
            preds: (B, V, lat, lon) where V includes all variables
            target: (B, V, lat, lon) where V includes all variables
        """
        # Extract only predictive variables
        preds_pred = preds[:, self.predictive_indices, :, :]
        target_pred = target[:, self.predictive_indices, :, :]

        self.rmse.update(preds_pred, target_pred)
        self.mae.update(preds_pred, target_pred)

        self.overall_mse.update(preds_pred.flatten(), target_pred.flatten())
        self.overall_mae.update(preds_pred.flatten(), target_pred.flatten())

    def compute(self) -> dict[str, float]:
        """Compute all metrics and return as dict."""
        metrics = {}

        # Per-variable RMSE and MAE
        rmse_values = torch.sqrt(self.rmse.compute())
        mae_values = self.mae.compute()

        for i, var_name in enumerate(self.predictive_variables):
            metrics[f'rmse/{var_name}'] = rmse_values[i].item()
            metrics[f'mae/{var_name}'] = mae_values[i].item()

        # Overall metrics
        metrics['overall/rmse'] = torch.sqrt(self.overall_mse.compute()).item()
        metrics['overall/mae'] = self.overall_mae.compute().item()

        return metrics

    def reset(self):
        """Reset all metrics."""
        self.rmse.reset()
        self.mae.reset()
        self.overall_mse.reset()
        self.overall_mae.reset()
