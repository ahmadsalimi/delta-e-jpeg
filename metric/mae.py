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


class WorstMAE(nn.Module):

    def __init__(self, th: float = 0.1):
        super().__init__()
        self.th = th

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor, *_, **__) -> torch.Tensor:
        """Compute the mean absolute error.

        Args:
            x (torch.Tensor): The original images with shape :math:`(B, C, H, W)`.
            x_hat (torch.Tensor): The reconstructed images with shape :math:`(B, C, H, W)`.
        """
        mae = (x - x_hat).abs().mean(dim=1)     # B x H x W
        return mae[mae > self.th].mean()
