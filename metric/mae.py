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


class LowPassMAE(nn.Module):

    def forward(self, x: torch.Tensor, x_hp: torch.Tensor, x_lp: torch.Tensor, *_, **__) -> torch.Tensor:
        """Compute the mean absolute error.

        Args:
            x (torch.Tensor): The original images with shape :math:`(B, C, H, W)`.
            x_hp (torch.Tensor): The high-pass filtered images with shape :math:`(B, C, H, W)`.
            x_lp (torch.Tensor): The low-pass predicted images with shape :math:`(B, C, H, W)`.
        """
        x_lp_true = x - x_hp
        return (x_lp_true - x_lp).abs().mean()
