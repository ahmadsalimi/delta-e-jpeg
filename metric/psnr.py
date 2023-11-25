import torch
from torch import nn


class PSNR(nn.Module):
    """Compute the PSNR between two batches of images.

    Args:
        data_range: The range of the data. Defaults to 1.
    """

    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor, *_, **__) -> torch.Tensor:
        """Compute the PSNR between two batches of images.

        Args:
            x (torch.Tensor): A batch of images with shape :math:`(B, C, H, W)`.
            x_hat (torch.Tensor): A batch of images with shape :math:`(B, C, H, W)`.
        """
        mse = ((x - x_hat).pow(2)).mean(dim=(-3, -2, -1))
        return (10 * torch.log10(self.data_range ** 2 / (mse + 1e-8))).mean()
