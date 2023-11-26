from typing import Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F

from jpeg.utils import Clamp, pad_or_crop, check_nan, NanChecker


class ConvDownsample(nn.Module):
    """Downsample a batch of images using convolution.

    Args:
        channels (int): The number of intermediate channels.
        kernel_size (int): The size of the convolution kernel. Defaults to 2.
        factor (Tuple[int, int]): The downsampling factor along the height and width dimension.
            For instance, for 4:2:0 downsampling, use ``factor=(2, 2)``, for 4:2:2 downsampling,
            use ``factor=(1, 2)``. Defaults to (2, 2).
    """

    def __init__(self, channels: int, kernel_size: int = 2,
                 factor: Tuple[int, int] = (2, 2), init: bool = True,
                 clamp: bool = True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv = nn.Sequential(                                                      # B x 2 x H x W
            nn.Conv2d(2, channels, kernel_size,
                      stride=factor, padding=((kernel_size - factor[0] + 1) // 2,
                                              (kernel_size - factor[1] + 1) // 2)),     # B x C x H/f x W/f
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, 1),                            # B x 2 x H/f x W/f
            *([] if not clamp else [
                Clamp(-0.5, 0.5),
            ]),
        )
        if init:
            self._init_weights()

    def _init_weights(self) -> None:
        """Initialize the weights of the convolutional layers to averaging"""
        # self.conv[0].weight.data[:self.channels // 2, 0].normal_(1 / self.kernel_size ** 2, 0.05)
        # self.conv[0].weight.data[:self.channels // 2, 1].normal_(0, 0.05)
        # self.conv[0].weight.data[self.channels // 2:, 0].normal_(0, 0.05)
        # self.conv[0].weight.data[self.channels // 2:, 1].normal_(1 / self.kernel_size ** 2, 0.05)
        # self.conv[0].bias.data.fill_(0.5)
        #
        # self.conv[2].weight.data[0, :self.channels // 2].normal_(1 / (self.channels // 2), 0.05)
        # self.conv[2].weight.data[1, :self.channels // 2].normal_(0, 0.05)
        # self.conv[2].weight.data[0, self.channels // 2:].normal_(0, 0.05)
        # self.conv[2].weight.data[1, self.channels // 2:].normal_(1 / (self.channels - self.channels // 2), 0.05)
        # self.conv[2].bias.data.fill_(-0.5)
        self.conv[0].weight.data[:self.channels // 2, 0].fill_(0)
        self.conv[0].weight.data[:self.channels // 2, 0, 0, 0].fill_(1)
        self.conv[0].weight.data[:self.channels // 2, 1].fill_(0)
        self.conv[0].weight.data[self.channels // 2:, 0].fill_(0)
        self.conv[0].weight.data[self.channels // 2:, 1].fill_(0)
        self.conv[0].weight.data[self.channels // 2:, 1, 0, 0].fill_(1)
        self.conv[0].bias.data.fill_(0.5)

        self.conv[2].weight.data[0, :self.channels // 2].fill_(1 / (self.channels // 2))
        self.conv[2].weight.data[1, :self.channels // 2].fill_(0)
        self.conv[2].weight.data[0, self.channels // 2:].fill_(0)
        self.conv[2].weight.data[1, self.channels // 2:].fill_(1 / (self.channels - self.channels // 2))
        self.conv[2].bias.data.fill_(-0.5)

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
        check_nan(y, 'downsample->y')
        cbcr = ycbcr[:, 1:]                                     # B x 2 x H x W
        check_nan(cbcr, 'downsample->cbcr')
        cbcr = self.conv(cbcr)                                  # B x 2 x H/f x W/f
        check_nan(cbcr, 'downsample->conv')
        return y, cbcr


class OtherUpsample(nn.Module):
    """Upsample a batch of images using convolution.

    Args:
        channels (int): The number of intermediate channels.
        kernel_size (int): The size of the convolution kernel. Defaults to 2.
        factor (Tuple[int, int]): The upsampling factor along the height and width dimension.
            For instance, for 4:2:0 upsampling, use ``factor=(2, 2)``, for 4:2:2 upsampling,
            use ``factor=(1, 2)``. Defaults to (2, 2).
        final_kernel_size (Optional[int]): The size of the final convolution kernel. If ``None``,
            no final convolution is applied. Defaults to 3.
    """

    def __init__(self, channels: int, final_kernel_size: int = 3, clamp: bool = True):
        super().__init__()
        self.channels = channels
        self.final_kernel_size_ = final_kernel_size
        self.conv = nn.Sequential(                                                  # B x 2 x H x W
            nn.Conv2d(2, channels, final_kernel_size, padding=(final_kernel_size - 1) // 2),  # B x C x H x W
            nn.Dropout2d(0.5, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, 1),                            # B x 2 x H x W
            *([] if not clamp else [
                Clamp(-0.5, 0.5),
                NanChecker('upsample->conv->clamp'),
            ]),
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
        cbcr = F.interpolate(cbcr, size=y.shape[-2:], mode='nearest')
        cbcr = self.conv(cbcr)                                  # B x 2 x H x W
        return torch.cat([y, cbcr], dim=1)


class ConvUpsample(nn.Module):
    """Upsample a batch of images using convolution.

    Args:
        channels (int): The number of intermediate channels.
        kernel_size (int): The size of the convolution kernel. Defaults to 2.
        factor (Tuple[int, int]): The upsampling factor along the height and width dimension.
            For instance, for 4:2:0 upsampling, use ``factor=(2, 2)``, for 4:2:2 upsampling,
            use ``factor=(1, 2)``. Defaults to (2, 2).
        final_kernel_size (Optional[int]): The size of the final convolution kernel. If ``None``,
            no final convolution is applied. Defaults to 3.
    """

    def __init__(self, channels: int, kernel_size: int = 2, factor: Tuple[int, int] = (2, 2),
                 final_kernel_size: Optional[int] = 3, init: bool = True, clamp: bool = True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.final_kernel_size_ = final_kernel_size
        self.conv = nn.Sequential(                                                  # B x 2 x H/f x W/f
            nn.Conv2d(2, channels, 1),                         # B x C x H/f x W/f
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            NanChecker('upsample->conv->relu'),
            *([
                  nn.ConvTranspose2d(channels, 2, kernel_size,          # B x 2 x H x W
                                     stride=factor, padding=((kernel_size - factor[0]) // 2,
                                                             (kernel_size - factor[1]) // 2)),
                  NanChecker('upsample->conv->2'),
              ] if final_kernel_size is None else [
                nn.ConvTranspose2d(channels, channels, kernel_size,                 # B x C x H x W
                                   stride=factor, padding=((kernel_size - factor[0]) // 2,
                                                           (kernel_size - factor[1]) // 2)),
                nn.Dropout2d(0.5),
                NanChecker('upsample->conv->2'),
                nn.ReLU(inplace=True),
                NanChecker('upsample->conv->relu2'),
                nn.Conv2d(channels, 2, final_kernel_size,               # B x 2 x H x W
                          padding=(final_kernel_size - 1) // 2),
                NanChecker('upsample->conv->3'),
            ]),
            *([] if not clamp else [
                Clamp(-0.5, 0.5),
                NanChecker('upsample->conv->clamp'),
            ]),
        )
        if init:
            self._init_weights()

    def _init_weights(self) -> None:
        """Initialize the weights of the convolutional layers to averaging"""
        # self.conv[0].weight.data[:self.channels // 2, 0].normal_(1, 0.01)
        # self.conv[0].weight.data[:self.channels // 2, 1].normal_(0, 0.01)
        # self.conv[0].weight.data[self.channels // 2:, 0].normal_(0, 0.01)
        # self.conv[0].weight.data[self.channels // 2:, 1].normal_(1, 0.01)
        # self.conv[0].bias.data.fill_(0.5)
        #
        # if self.final_kernel_size_ is None:
        #     self.conv[2].weight.data[:self.channels // 2, 0].normal_(1 / (self.channels // 2), 0.01)
        #     self.conv[2].weight.data[:self.channels // 2, 1].normal_(0, 0.01)
        #     self.conv[2].weight.data[self.channels // 2:, 0].normal_(0, 0.01)
        #     self.conv[2].weight.data[self.channels // 2:, 1].normal_(1 / (self.channels - self.channels // 2), 0.01)
        #     self.conv[2].bias.data.fill_(-0.5)
        # else:
        #     for i in range(self.channels):
        #         self.conv[2].weight.data[i, i].normal_(1, 0.01)
        #         self.conv[2].weight.data[i, :i].normal_(0, 0.01)
        #         self.conv[2].weight.data[i, i + 1:].normal_(0, 0.01)
        #     self.conv[2].bias.data.zero_()
        #
        #     self.conv[4].weight.data[0, :self.channels // 2].normal_(0, 0.01)
        #     self.conv[4].weight\
        #         .data[0, :self.channels // 2, self.final_kernel_size_ // 2, self.final_kernel_size_ // 2].normal_(
        #         1 / (self.channels // 2), 0.01)
        #     self.conv[4].weight.data[1, :self.channels // 2].normal_(0, 0.01)
        #     self.conv[4].weight.data[0, self.channels // 2:].normal_(0, 0.01)
        #     self.conv[4].weight.data[1, self.channels // 2:].normal_(0, 0.01)
        #     self.conv[4].weight\
        #         .data[1, self.channels // 2:, self.final_kernel_size_ // 2, self.final_kernel_size_ // 2].normal_(
        #         1 / (self.channels - self.channels // 2), 0.01)
        #     self.conv[4].bias.data.fill_(-0.5)
        self.conv[0].weight.data[:self.channels // 2, 0].fill_(1)
        self.conv[0].weight.data[:self.channels // 2, 1].fill_(0)
        self.conv[0].weight.data[self.channels // 2:, 0].fill_(0)
        self.conv[0].weight.data[self.channels // 2:, 1].fill_(1)
        self.conv[0].bias.data.fill_(0.5)

        if self.final_kernel_size_ is None:
            self.conv[4].weight.data[:self.channels // 2, 0].fill_(1 / (self.channels // 2))
            self.conv[4].weight.data[:self.channels // 2, 1].fill_(0)
            self.conv[4].weight.data[self.channels // 2:, 0].fill_(0)
            self.conv[4].weight.data[self.channels // 2:, 1].fill_(1 / (self.channels - self.channels // 2))
            self.conv[4].bias.data.fill_(-0.5)
        else:
            for i in range(self.channels):
                self.conv[4].weight.data[i, i].fill_(1)
                self.conv[4].weight.data[i, :i].fill_(0)
                self.conv[4].weight.data[i, i + 1:].fill_(0)
            self.conv[4].bias.data.zero_()

            self.conv[8].weight.data[0, :self.channels // 2].fill_(0)
            self.conv[8].weight\
                .data[0, :self.channels // 2, self.final_kernel_size_ // 2, self.final_kernel_size_ // 2].fill_(
                1 / (self.channels // 2))
            self.conv[8].weight.data[1, :self.channels // 2].fill_(0)
            self.conv[8].weight.data[0, self.channels // 2:].fill_(0)
            self.conv[8].weight.data[1, self.channels // 2:].fill_(0)
            self.conv[8].weight\
                .data[1, self.channels // 2:, self.final_kernel_size_ // 2, self.final_kernel_size_ // 2].fill_(
                1 / (self.channels - self.channels // 2))
            self.conv[8].bias.data.fill_(-0.5)

    def forward(self, y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
        """Upsample a batch of images.

        Args:
            y (torch.Tensor): A batch of luminance channels to upsample with shape :math:`(B, 1, H, W)`.
            cbcr (torch.Tensor): A batch of chrominance channels to upsample with shape
                :math:`(B, 2, H / factor[0], W / factor[1])`.

        Returns:
            torch.Tensor: The upsampled images with shape :math:`(B, 3, H, W)`.
        """
        check_nan(y, 'upsample->y')
        check_nan(cbcr, 'upsample->cbcr')
        try:
            cbcr = self.conv(cbcr)                                  # B x 2 x H x W
        except:
            print(self.conv[0].weight.data.min(), self.conv[0].weight.data.max(),
                  self.conv[0].bias.data.min(), self.conv[0].bias.data.max())
            raise
        check_nan(cbcr, 'upsample->conv')
        cbcr = pad_or_crop(cbcr, y.shape[-2:])                  # B x 2 x H x W
        check_nan(cbcr, 'upsample->pad_or_crop')
        return torch.cat([y, cbcr], dim=1)
