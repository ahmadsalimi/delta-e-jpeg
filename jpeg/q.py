from typing import Literal

import torch
from torch import nn

# Quantization tables
Q_y = torch.tensor([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])

Q_c = torch.tensor([[17, 18, 24, 47, 99, 99, 99, 99],
                    [18, 21, 26, 66, 99, 99, 99, 99],
                    [24, 26, 56, 99, 99, 99, 99, 99],
                    [47, 66, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99]])


def __apply_quality(Q: torch.Tensor, quality: int = 50) -> torch.Tensor:
    """Get the quantization table for a given quality factor.

    Args:
        Q (torch.Tensor): The quantization table with shape :math:`(M, N)`.
        quality (int): The quality factor.

    Returns:
        torch.Tensor: The quantization table.
    """
    assert 1 <= quality <= 100, f'Quality factor must be in [1, 100], got {quality}'
    s = 5000 / quality if quality < 50 else 200 - 2 * quality
    Q = ((s * Q + 50) / 100).floor().clamp(1)
    return Q


def q(x: torch.Tensor, Q: torch.Tensor, quality: int = 50) -> torch.Tensor:
    """Quantize a channel.

    Args:
        x (torch.Tensor): The channel to quantize with shape :math:`(*, M, N)`.
        Q (torch.Tensor): The quantization table with shape :math:`(M, N)`.
        quality (float): The quality factor. Defaults to 50.

    Returns:
        torch.Tensor: The quantized channel with shape :math:`(*, M, N)`.
    """
    Q = __apply_quality(Q, quality).to(x.device)
    return (x * 100 / Q).round()       # * x M x N


def iq(x: torch.Tensor, Q: torch.Tensor, quality: int = 50) -> torch.Tensor:
    """Inverse quantize a channel.

    Args:
        x (torch.Tensor): The channel to inverse quantize with shape :math:`(*, M, N)`.
        Q (torch.Tensor): The quantization table with shape :math:`(M, N)`.
        quality (float): The quality factor. Defaults to 50.

    Returns:
        torch.Tensor: The inverse quantized channel with shape :math:`(*, M, N)`.
    """
    Q = __apply_quality(Q, quality).to(x.device)
    return x * Q / 100                  # * x M x N


def q_y(y: torch.Tensor) -> torch.Tensor:
    """Quantize the luminance channel.

    Args:
        y (torch.Tensor): The luminance channel to quantize with shape :math:`(*, 8, 8)`.

    Returns:
        torch.Tensor: The quantized luminance channel with shape :math:`(*, 8, 8)`.
    """
    return q(y, Q_y)


def q_c(c: torch.Tensor) -> torch.Tensor:
    """Quantize the chrominance channel.

    Args:
        c (torch.Tensor): The chrominance channel to quantize with shape :math:`(*, 8, *, 8, *)`.

    Returns:
        torch.Tensor: The quantized chrominance channel with shape :math:`(*, 8, *, 8, *)`.
    """
    return q(c, Q_c)


def iq_y(y: torch.Tensor) -> torch.Tensor:
    """Inverse quantize the luminance channel.

    Args:
        y (torch.Tensor): The luminance channel to inverse quantize with shape :math:`(*, 8, *, 8, *)`.

    Returns:
        torch.Tensor: The inverse quantized luminance channel with shape :math:`(*, 8, *, 8, *)`.
    """
    return iq(y, Q_y)


def iq_c(c: torch.Tensor) -> torch.Tensor:
    """Inverse quantize the chrominance channel.

    Args:
        c (torch.Tensor): The chrominance channel to inverse quantize with shape :math:`(*, 8, *, 8, *)`.

    Returns:
        torch.Tensor: The inverse quantized chrominance channel with shape :math:`(*, 8, *, 8, *)`.
    """
    return iq(c, Q_c)


class Quantizer:
    """Quantize and inverse quantize a channel.

    Args:
        mode (Literal['y', 'c']): The channel to quantize, either luminance ('y') or chrominance ('c').
        quality (int): The quality factor. Defaults to 50.
    """

    def __init__(self, mode: Literal['y', 'c'], quality: int = 50):
        self.Q = Q_y if mode == 'l' else Q_c
        self.quality = quality

    def q(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize a channel.

        Args:
            x (torch.Tensor): The channel to quantize with shape :math:`(*, M, N)`.

        Returns:
            torch.Tensor: The quantized channel with shape :math:`(*, M, N)`.
        """
        return q(x, self.Q, self.quality)

    def iq(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse quantize a channel.

        Args:
            x (torch.Tensor): The channel to inverse quantize with shape :math:`(*, M, N)`.

        Returns:
            torch.Tensor: The inverse quantized channel with shape :math:`(*, M, N)`.
        """
        return iq(x, self.Q, self.quality)
