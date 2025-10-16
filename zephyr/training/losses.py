import numpy as np
import torch
from torch import nn


class LatitudeWeightedMSELoss(nn.Module):
    """Latitude-weighted MSE loss for weather forecasting."""

    def __init__(self, variables: list[str], forced: list[str] | None = None) -> None:
        super().__init__()
        self.loss_variables = (
            [i for i, v in enumerate(variables) if v not in forced]
            if forced is not None
            else list(range(len(variables)))
        )
        self.loss_variables_names = np.array(list(variables))
        self.loss_variables_names = self.loss_variables_names[self.loss_variables]
        self.loss = nn.MSELoss(reduce=None)

    def forward(
        self, true: torch.Tensor, pred: torch.Tensor, return_per_variable: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Compute MSE loss, optionally returning per-variable breakdown.

        Args:
            true: Ground truth tensor of shape (B, V, H, W)
            pred: Predictions tensor of shape (B, V, H, W)
            return_per_variable: If True, return dict with total loss and per-variable losses

        Returns:
            If return_per_variable=False: Single loss value (scalar tensor)
            If return_per_variable=True: Dict with 'total_loss' and per-variable losses
        """
        true = true[:, self.loss_variables, ...]  # B x V x H x W
        pred = pred[:, self.loss_variables, ...]  # B x V x H x W
        mse = self.loss(true, pred)  # B x V x H x W

        total_loss = mse.mean()

        if return_per_variable:
            # Compute per-variable loss: average over batch, H, W dimensions
            per_var_loss = mse.mean(dim=(0, 2, 3))  # (V,)

            # Create dict with variable names
            loss_dict = {
                "total_loss": total_loss,
                **{
                    f"loss/{var_name}": loss_value.item()
                    for var_name, loss_value in zip(self.loss_variables_names, per_var_loss)
                }
            }
            return loss_dict

        return total_loss
