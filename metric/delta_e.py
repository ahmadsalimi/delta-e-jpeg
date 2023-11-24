import torch
from torch import nn
import kornia as K


class DeltaE(nn.Module):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the delta E between two batches of images.

        Args:
            x (torch.Tensor): A batch of images in RGB format with shape :math:`(B, 3, H, W)`.
                The images are assumed to be in the range :math:`[0, 1]`.
            y (torch.Tensor): A batch of images in RGB format with shape :math:`(B, 3, H, W)`.
                The images are assumed to be in the range :math:`[0, 1]`.

        Returns:
            torch.Tensor: The delta E between the two batches of images with shape :math:`(B)`.
        """
        x = K.color.rgb_to_lab(x)
        y = K.color.rgb_to_lab(y)
        return (x - y).norm(dim=1).mean()
