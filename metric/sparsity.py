import torch
from torch import nn


class Sparsity(nn.Module):

    def forward(self, y: torch.Tensor, cbcr: torch.Tensor, *_, **__) -> torch.Tensor:
        """Compute the sparsity.

        Args:
            y (torch.Tensor): The luminance channel with shape :math:`(B, 1, H, W)`.
            cbcr (torch.Tensor): The chrominance channels with shape :math:`(B, 2, H, W)`.
        """
        return ((y == 0).sum() + (cbcr == 0).sum()) / (y.numel() + cbcr.numel())
