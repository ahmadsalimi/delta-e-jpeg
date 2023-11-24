from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class Clamp(nn.Module):

    def __init__(self, min_: float = 0.0, max_: float = 1.0):
        super().__init__()
        self.min = min_
        self.max = max_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(self.min, self.max)


class ConvDownsample(nn.Module):
    """Downsample a batch of images using convolution.

    Args:
        channels (int): The number of intermediate channels.
        kernel_size (int): The size of the convolution kernel. Defaults to 2.
        factor (Tuple[int, int]): The downsampling factor along the height and width dimension.
            For instance, for 4:2:0 downsampling, use ``factor=(2, 2)``, for 4:2:2 downsampling,
            use ``factor=(1, 2)``. Defaults to (2, 2).
    """

    def __init__(self, channels: int, kernel_size: int = 2, factor: Tuple[int, int] = (2, 2)):
        super().__init__()
        self.conv = nn.Sequential(                                                  # B x 2 x H x W
            nn.Conv2d(2, channels, kernel_size,
                      stride=factor, padding=(kernel_size - factor[0] + 1) // 2),   # B x C x H/f x W/f
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, 1),                        # B x 2 x H/f x W/f
            Clamp(0, 1),
        )

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
        cbcr = self.conv(cbcr)                                  # B x 2 x H/f x W/f
        return y, cbcr


class ConvUpsample(nn.Module):
    """Upsample a batch of images using convolution.

    Args:
        channels (int): The number of intermediate channels.
        kernel_size (int): The size of the convolution kernel. Defaults to 2.
        factor (Tuple[int, int]): The upsampling factor along the height and width dimension.
            For instance, for 4:2:0 upsampling, use ``factor=(2, 2)``, for 4:2:2 upsampling,
            use ``factor=(1, 2)``. Defaults to (2, 2).
    """

    def __init__(self, channels: int, kernel_size: int = 2, factor: Tuple[int, int] = (2, 2)):
        super().__init__()
        self.conv = nn.Sequential(                                                  # B x 2 x H/f x W/f
            nn.Conv2d(2, channels, 1),                         # B x C x H/f x W/f
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels, 2, kernel_size,                # B x 2 x H x W
                               stride=factor, padding=(kernel_size - factor[0]) // 2),
            Clamp(0, 1),
        )

    def forward(self, y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
        """Upsample a batch of images.

        Args:
            y (torch.Tensor): A batch of luminance channels to upsample with shape :math:`(B, 1, H, W)`.
            cbcr (torch.Tensor): A batch of chrominance channels to upsample with shape
                :math:`(B, 2, H / factor[0], W / factor[1])`.

        Returns:
            torch.Tensor: The upsampled images with shape :math:`(B, 3, H, W)`.
        """
        cbcr = self.conv(cbcr)                                  # B x 2 x H x W
        if cbcr.shape[-2] > y.shape[-2]:
            cbcr = cbcr[..., :y.shape[-2]-cbcr.shape[-2], :]
        elif cbcr.shape[-2:] < y.shape[-2:]:
            cbcr = F.pad(cbcr, (0, 0, 0, y.shape[-2] - cbcr.shape[-2]))
        if cbcr.shape[-1] > y.shape[-1]:
            cbcr = cbcr[..., :y.shape[-1]-cbcr.shape[-1]]
        elif cbcr.shape[-1] < y.shape[-1]:
            cbcr = F.pad(cbcr, (0, y.shape[-1] - cbcr.shape[-1]))
        assert cbcr.shape[-2:] == y.shape[-2:], f"{cbcr.shape[-2:]} != {y.shape[-2:]}"
        return torch.cat([y, cbcr], dim=1)
