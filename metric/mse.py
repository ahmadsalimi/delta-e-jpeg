import torch
from torch import nn


class MSE(nn.Module):

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor, *_, **__) -> torch.Tensor:
        """Compute the MSE between two batches of images.

        Args:
            x (torch.Tensor): A batch of images with shape :math:`(B, C, H, W)`.
            x_hat (torch.Tensor): A batch of images with shape :math:`(B, C, H, W)`.
        """
        return (x - x_hat).pow(2).mean()
