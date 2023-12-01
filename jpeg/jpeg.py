from typing import Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
import kornia as K

from jpeg.dct import dct2d, idct2d
from jpeg.ds.naive import NaiveDownsample, NaiveUpsample
from jpeg.q import Quantizer


class ExtendedJPEG(nn.Module):

    def __init__(self, downsample: Optional[nn.Module] = None, upsample: Optional[nn.Module] = None,
                 quality: int = 50, lp_kernel_size: int = 11, lp_sigma: float = 3):
        super().__init__()
        self.block_size = 8
        # downsample cb and cr by 4:2:0
        self.downsample = downsample or NaiveDownsample(factor=(2, 2))
        self.upsample = upsample or NaiveUpsample()
        self.q_y = Quantizer('y', quality=quality)
        self.q_c = Quantizer('c', quality=quality)
        self.lp = K.filters.GaussianBlur2d((lp_kernel_size, lp_kernel_size), (lp_sigma, lp_sigma))

    def __zero_pad(self, image: torch.Tensor) -> torch.Tensor:
        """Pad a batch of images to be a multiple of the block size.

        Args:
            image (torch.Tensor): A batch of images to pad with shape :math:`(B, C, H, W)`.

        Returns:
            torch.Tensor: The padded images with shape :math:`(B, C, H + H % block_size, W + W % block_size)`.
        """
        return F.pad(image, (0, (-image.shape[-1]) % self.block_size,
                             0, (-image.shape[-2]) % self.block_size))

    def __split_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """Split a batch of images into blocks.

        Args:
            x (torch.Tensor): A batch of images to split with shape :math:`(B, C, H, W)`.

        Returns:
            torch.Tensor: The split images with shape
                :math:`(B, C, H / block_size, W / block_size, block_size, block_size)`.
        """
        x = self.__zero_pad(x)                                 # B x C x H x W
        B, C, H, W = x.shape
        N = self.block_size
        x = x.reshape(B, C, H // N, N, W // 8, N)       # B x C x H/8 x 8 x W/8 x 8
        x = x.permute(0, 1, 2, 4, 3, 5)                        # B x C x H/8 x W/8 x 8 x 8
        return x

    @staticmethod
    def __merge_blocks(x: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        """Merge a batch of images from blocks.

        Args:
            x (torch.Tensor): A batch of images to merge with shape
                :math:`(B, C, H / block_size, W / block_size, block_size, block_size)`.
            shape (Tuple[int, int]): The shape of the original images to remove zero-padding.

        Returns:
            torch.Tensor: The merged images with shape :math:`(B, C, H, W)`.
        """
        B, C, Hb, Wb, N, _ = x.shape
        x = x.permute(0, 1, 2, 4, 3, 5)       # B x C x H/8 x 8 x W/8 x 8
        x = x.reshape(B, C, Hb * N, Wb * N)         # B x C x H x W
        x = x[..., :shape[0], :shape[1]]            # B x C x H x W
        return x

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Encode and decode a batch of images in RGB space into and from JPEG format.

        Args:
            rgb (torch.Tensor): A batch of images in RGB space with shape :math:`(B, 3, H, W)`.
                The images are assumed to be in the range :math:`[0, 1]`.

        Returns:
            torch.Tensor: The reconstructed images in RGB space with shape :math:`(B, 3, H, W)`.
        """
        _, _, H, W = rgb.shape
        ycbcr = K.color.rgb_to_ycbcr(rgb) - 0.5     # B x 3 x H x W

        # downsample cb and cr
        y, cbcr = self.downsample(ycbcr)            # B x 1 x H x W, B x 2 x H/2 x W/2

        # apply a low-pass filter
        y = self.lp(y)                              # B x 1 x H x W
        cbcr = self.lp(cbcr)                        # B x 2 x H/2 x W/2

        # upsample cb and cr
        y = self.upsample(y, cbcr)                  # B x 3 x H x W

        # convert to rgb
        rgb = K.color.ycbcr_to_rgb(y + 0.5)         # B x 3 x H x W

        # clamp to [0, 1]
        rgb = torch.clamp(rgb, 0, 1)      # B x 3 x H x W

        return rgb

    def get_mapping(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of images in RGB space into JPEG format.

        Args:
            rgb (torch.Tensor): A batch of images to encode with shape :math:`(B, 3, H, W)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The encoded luminance and chrominance blocks.
                The luminance channel has shape :math:`(B, 1,
        """
        ycbcr = K.color.rgb_to_ycbcr(rgb) - 0.5     # B x 3 x H x W

        # downsample cb and cr
        y, cbcr = self.downsample(ycbcr)            # B x 1 x H x W, B x 2 x H/2 x W/2

        return y, cbcr

    def encode(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of images in RGB space into JPEG format.

        Args:
            rgb (torch.Tensor): A batch of images to encode with shape :math:`(B, 3, H, W)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The encoded luminance and chrominance blocks.
                The luminance channel has shape :math:`(B, 1,
        """
        ycbcr = K.color.rgb_to_ycbcr(rgb) - 0.5     # B x 3 x H x W

        # downsample cb and cr
        y, cbcr = self.downsample(ycbcr)            # B x 1 x H x W, B x 2 x H/2 x W/2

        # split into blocks
        y = self.__split_blocks(y)                  # B x 1 x H/8 x W/8 x 8 x 8
        cbcr = self.__split_blocks(cbcr)            # B x 2 x H/16 x W/16 x 8 x 8

        # apply dct
        y = dct2d(y)                                # B x 1 x H/8 x W/8 x 8 x 8
        cbcr = dct2d(cbcr)                          # B x 2 x H/16 x W/16 x 8 x 8

        # quantize
        y = self.q_y.q(y)                                  # B x 1 x H/8 x W/8 x 8 x 8
        cbcr = self.q_c.q(cbcr)                            # B x 2 x H/16 x W/16 x 8 x 8

        return y, cbcr

    def decode(self, y: torch.Tensor, cbcr: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        """Decode a batch of images from JPEG format into RGB space.

        Args:
            y (torch.Tensor): The encoded luminance blocks with shape :math:`(B, 1 x H/8 x W/8 x 8 x 8)`.
            cbcr (torch.Tensor): The encoded chrominance blocks with shape :math:`(B, 2 x H/16 x W/16 x 8 x 8)`.
            shape (Tuple[int, int]): The shape of the original images to remove zero-padding.

        Returns:
            torch.Tensor: The decoded images in RGB space with shape :math:`(B, 3, H, W)`.
        """
        # dequantize
        y = self.q_y.iq(y)                                 # B x 1 x H/8 x W/8 x 8 x 8
        cbcr = self.q_c.iq(cbcr)                           # B x 2 x H/16 x W/16 x 8 x 8

        # apply idct
        y = idct2d(y)                               # B x 1 x H/8 x W/8 x 8 x 8
        cbcr = idct2d(cbcr)                         # B x 2 x H/16 x W/16 x 8 x 8

        # merge blocks
        y = self.__merge_blocks(y, shape)           # B x 1 x H x W
        cbcr = self.__merge_blocks(cbcr, (shape[0] // 2, shape[1] // 2))  # B x 2 x H/2 x W/2

        # upsample cb and cr
        ycbcr = self.upsample(y, cbcr)              # B x 3 x H x W

        # convert to rgb
        rgb = K.color.ycbcr_to_rgb(ycbcr + 0.5)     # B x 3 x H x W

        # clamp to [0, 1]
        rgb = torch.clamp(rgb, 0, 1)      # B x 3 x H x W
        return rgb
