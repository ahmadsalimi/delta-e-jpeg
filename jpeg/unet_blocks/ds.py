import torch
from torch import nn

from jpeg.unet_blocks.res import ResidualBlock


class DownBlock(nn.Module):
    """
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py
    ### Down block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, n_groups: int = 32):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, n_groups=n_groups)

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        return x


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)
