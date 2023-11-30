from typing import Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from jpeg.unet_blocks.ds import DownBlock, Downsample
from jpeg.unet_blocks.mid import MiddleBlock
from jpeg.unet_blocks.swish import Swish
from jpeg.unet_blocks.us import UpBlock, Upsample


class UNetUpsample(nn.Module):
    """
    ## U-Net
    """

    def __init__(self, image_channels: int = 2, n_channels: int = 64,
                 ch_mults: Sequence[int] = (1, 2),
                 n_blocks: int = 2,
                 n_groups: int = 32):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        self.n_resolutions = n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_groups=n_groups))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_groups=n_groups)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_groups=n_groups))
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_groups=n_groups))
            in_channels = out_channels
            # Up sample at all resolutions except last
            # if i > 0:
            up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

    def _forward(self, x: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        """

        # Get image projection
        x = self.image_proj(x)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x)

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                #
                x = m(x)

        # Final normalization and convolution
        return self.final(self.act(self.norm(x)))

    def forward(self, y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
        n_ds = self.n_resolutions - 1
        # pad to be divisible by 2**n_ds
        H, W = cbcr.shape[-2:]
        H_pad = (-H) % (2 ** n_ds)
        W_pad = (-W) % (2 ** n_ds)
        padding = (W_pad // 2, W_pad - W_pad // 2, H_pad // 2, H_pad - H_pad // 2)
        cbcr = F.pad(cbcr, padding)

        cbcr = self._forward(cbcr)
        if y.shape[-2:] != cbcr.shape[-2:]:
            cbcr = F.interpolate(cbcr, size=y.shape[-2:], mode='bilinear', align_corners=True)
        return torch.cat((y, cbcr), dim=1)


class FullUNetUpsample(nn.Module):
    """
    ## Full U-Net
    """

    def __init__(self, n_channels: int = 64,
                 ch_mults: Sequence[int] = (1, 2),
                 n_blocks: int = 2,
                 n_groups: int = 32):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        self.n_resolutions = n_resolutions = len(ch_mults)
        self.n_blocks = n_blocks

        # Project image into feature map
        self.image_proj = nn.Conv2d(1, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels + 2 * (i == 1), out_channels, n_groups=n_groups))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_groups=n_groups)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for j in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_groups=n_groups))
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels + 2 * (i == 1), out_channels, n_groups=n_groups))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, 3, kernel_size=(3, 3), padding=(1, 1))

    def _forward(self, y: torch.Tensor, cbcr: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        """

        # Get image projection
        x = self.image_proj(y)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for i, m in enumerate(self.down):
            x = m(x)
            if i == self.n_blocks:
                x = torch.cat((x, cbcr), dim=1)
            h.append(x)
        # Middle (bottom)
        x = self.middle(x)

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                #
                x = m(x)

        # Final normalization and convolution
        return self.final(self.act(self.norm(x)))

    def forward(self, y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
        n_ds = self.n_resolutions - 1
        # pad to be divisible by 2**n_ds
        H, W = cbcr.shape[-2:]
        H_pad = (-H) % (2 ** n_ds)
        W_pad = (-W) % (2 ** n_ds)
        padding = (W_pad // 2, W_pad - W_pad // 2, H_pad // 2, H_pad - H_pad // 2)
        cbcr = F.pad(cbcr, padding)

        ycbcr = self._forward(y, cbcr)
        # if y.shape[-2:] != cbcr.shape[-2:]:
        #     cbcr = F.interpolate(cbcr, size=y.shape[-2:], mode='bilinear', align_corners=True)
        return ycbcr


class HighPassUNetUpsample(nn.Module):
    """
    ## High-Pass U-Net
    """

    def __init__(self, n_channels: int = 64,
                 ch_mults: Sequence[int] = (1, 2),
                 n_blocks: int = 2,
                 n_groups: int = 32):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        self.n_resolutions = n_resolutions = len(ch_mults)
        self.n_blocks = n_blocks

        # Project image into feature map
        self.image_proj = nn.Conv2d(3, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_groups=n_groups))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_groups=n_groups)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for j in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_groups=n_groups))
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_groups=n_groups))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, 3, kernel_size=(3, 3), padding=(1, 1))

    def _forward(self, x: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        """

        # Get image projection
        x = self.image_proj(x)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for i, m in enumerate(self.down):
            x = m(x)
            h.append(x)
        # Middle (bottom)
        x = self.middle(x)

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                #
                x = m(x)

        # Final normalization and convolution
        return self.final(self.act(self.norm(x)))

    def forward(self, y: torch.Tensor, cbcr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cbcr = F.interpolate(cbcr, size=y.shape[-2:], mode='bilinear', align_corners=True)
        ycbcr_lp = torch.cat((y, cbcr), dim=1)
        ycbcr_hp = self._forward(ycbcr_lp)
        # if y.shape[-2:] != cbcr.shape[-2:]:
        #     cbcr = F.interpolate(cbcr, size=y.shape[-2:], mode='bilinear', align_corners=True)
        return ycbcr_lp, ycbcr_hp
