import torch
from torch import nn
import lpips


class LPIPS(nn.Module):

    def __init__(self, net: str = 'alex'):
        super().__init__()
        self.lpips = lpips.LPIPS(net=net)

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor, *_, **__) -> torch.Tensor:
        """Compute the LPIPS between two batches of images.

        Args:
            x (torch.Tensor): A batch of images in RGB format with shape :math:`(B, 3, H, W)`.
                The images are assumed to be in the range :math:`[0, 1]`.
            x_hat (torch.Tensor): A batch of images in RGB format with shape :math:`(B, 3, H, W)`.
                The images are assumed to be in the range :math:`[0, 1]`.

        Returns:
            torch.Tensor: The LPIPS between the two batches of images with shape :math:`(B)`.
        """
        return self.lpips(x * 2 - 1, x_hat * 2 - 1).mean()
