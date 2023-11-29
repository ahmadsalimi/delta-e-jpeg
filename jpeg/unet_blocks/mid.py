import torch
from torch import nn

from jpeg.unet_blocks.res import ResidualBlock


class MiddleBlock(nn.Module):
    """
    ### Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, n_groups: int = 32):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, n_groups=n_groups)

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        return x
