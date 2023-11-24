import warnings
from typing import Tuple, Optional

import torch
from torch import nn

from jpeg.utils import Clamp, pad_or_crop


class FullConvDownsample(nn.Module):
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
        self.conv = nn.Sequential(                                                  # B x 3 x H x W
            nn.Conv2d(3, channels, kernel_size,
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
        cbcr = self.conv(ycbcr)                                 # B x 2 x H/f x W/f
        return y, cbcr


class FullConvUpsample(nn.Module):
    """Upsample a batch of images using convolution.

    Args:
        channels (int): The number of intermediate channels.
        kernel_size (int): The size of the convolution kernel. Defaults to 2.
        factor (Tuple[int, int]): The upsampling factor along the height and width dimension.
            For instance, for 4:2:0 upsampling, use ``factor=(2, 2)``, for 4:2:2 upsampling,
            use ``factor=(1, 2)``. Defaults to (2, 2).
        final_kernel_size (Optional[int]): The size of the final convolution kernel. If ``None``,
            no final convolution is applied. Defaults to 3.
        reconstruct_y (bool): Whether to reconstruct the luminance channel from the luminance
            and chrominance channels. Defaults to False.
    """

    def __init__(self, channels: int, kernel_size: int = 2, factor: Tuple[int, int] = (2, 2),
                 final_kernel_size: Optional[int] = 3, reconstruct_y: bool = False):
        if not final_kernel_size and reconstruct_y:
            warnings.warn("final_kernel_size is None, but reconstruct_y is True. "
                          "reconstruct_y will be ignored.")
        super().__init__()
        self.conv = nn.Sequential(                                                  # B x 2 x H/f x W/f
            nn.Conv2d(2, channels, 1),                         # B x C x H/f x W/f
            nn.ReLU(inplace=True),
            *([
                  nn.ConvTranspose2d(channels, 2, kernel_size,          # B x 2 x H x W
                                     stride=factor, padding=(kernel_size - factor[0]) // 2),
              ] if final_kernel_size is None else [
                nn.ConvTranspose2d(channels, channels, kernel_size,                 # B x C x H x W
                                   stride=factor, padding=(kernel_size - factor[0]) // 2),
            ]),
        )
        self.final_conv = nn.Sequential(
            *([] if final_kernel_size is None else [
                nn.ReLU(inplace=True),
                nn.Conv2d(channels + 1, 2 + reconstruct_y, final_kernel_size,       # B x {2,3} x H x W
                          padding=(final_kernel_size - 1) // 2),
            ]),
            Clamp(0, 1),
        )
        self.reconstruct_y = reconstruct_y

    def forward(self, y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
        """Upsample a batch of images.

        Args:
            y (torch.Tensor): A batch of luminance channels to upsample with shape :math:`(B, 1, H, W)`.
            cbcr (torch.Tensor): A batch of chrominance channels to upsample with shape
                :math:`(B, 2, H / factor[0], W / factor[1])`.

        Returns:
            torch.Tensor: The upsampled images with shape :math:`(B, 3, H, W)`.
        """
        cbcr = self.conv(cbcr)                                  # B x {2,C} x H x W
        cbcr = pad_or_crop(cbcr, y.shape[-2:])                  # B x {2,C} x H x W
        ycbcr = torch.cat([y, cbcr], dim=1)             # B x {3,C+1} x H x W
        if self.reconstruct_y:
            return self.final_conv(ycbcr)                       # B x 3 x H x W
        cbcr = self.final_conv(ycbcr)                           # B x 2 x H x W
        return torch.cat([y, cbcr], dim=1)              # B x 3 x H x W
