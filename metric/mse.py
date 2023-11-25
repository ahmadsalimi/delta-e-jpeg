import torch
from torch import nn


class MSE(nn.Module):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the MSE between two batches of images.

        Args:
            x (torch.Tensor): A batch of images with shape :math:`(B, C, H, W)`.
            y (torch.Tensor): A batch of images with shape :math:`(B, C, H, W)`.
        """
        return (x - y).pow(2).mean()
