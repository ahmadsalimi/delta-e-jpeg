import torch
from torch import nn


class MAE(nn.Module):

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor, *_, **__) -> torch.Tensor:
        """Compute the mean absolute error.

        Args:
            x (torch.Tensor): The original images with shape :math:`(B, C, H, W)`.
            x_hat (torch.Tensor): The reconstructed images with shape :math:`(B, C, H, W)`.
        """
        return (x - x_hat).abs().mean()
