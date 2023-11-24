from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class NaiveDownsample(nn.Module):
    """Naively downsample a batch of images in YCbCr format.

    Args:
        factor (Tuple[int, int]): The downsampling factor along the height and width dimension.
            For instance, for 4:2:0 downsampling, use ``factor=(2, 2)``, for 4:2:2 downsampling,
            use ``factor=(1, 2)``. Defaults to (2, 2).
    """

    def __init__(self, factor: Tuple[int, int] = (2, 2)):
        super().__init__()
        self.factor = factor

    def forward(self, ycbcr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Downsample a batch of images.

        Args:
            ycbcr (torch.Tensor): A batch of images to downsample with shape :math:`(B, 3, H, W)`.
                The images are assumed to be in YCbCr format.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The luminance and downsampled chrominance channels.
                The luminance channel has shape :math:`(B, 1, H, W)` and the chrominance channel
                has shape :math:`(B, 2, H / factor[0], W / factor[1])`.
        """
        y = ycbcr[:, :1]                                        # B x 1 x H x W
        cbcr = ycbcr[:, 1:]                                     # B x 2 x H x W
        cbcr = cbcr[..., ::self.factor[0], ::self.factor[1]]    # B x 2 x H / factor[0] x W / factor[1]
        return y, cbcr


class NaiveUpsample(nn.Module):

    def forward(self, y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
        """Upsample a batch of images.

        Args:
            y (torch.Tensor): A batch of luminance channels to upsample with shape :math:`(B, 1, H, W)`.
            cbcr (torch.Tensor): A batch of chrominance channels to upsample with shape
                :math:`(B, 2, H / factor[0], W / factor[1])`.

        Returns:
            torch.Tensor: The upsampled images with shape :math:`(B, 3, H, W)`.
        """
        print(y.shape, cbcr.shape)
        cbcr = F.interpolate(cbcr, size=y.shape[-2:], mode='bilinear', align_corners=True)
        return torch.cat([y, cbcr], dim=1)
