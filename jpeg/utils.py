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


def pad_or_crop(image: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Pad or crop an image to match the size of a base image.

    Args:
        image (torch.Tensor): The image to pad or crop with shape :math:`(B, C, H, W)`.
        shape (torch.Size): The shape of the base image.

    Returns:
        torch.Tensor: The padded or cropped image with shape :math:`(B, C, H, W)`.
    """
    if image.shape[-2] > shape[-2]:
        image = image[..., :shape[-2] - image.shape[-2], :]
    elif image.shape[-2:] < shape[-2:]:
        image = F.pad(image, (0, 0, 0, shape[-2] - image.shape[-2]))
    if image.shape[-1] > shape[-1]:
        image = image[..., :shape[-1] - image.shape[-1]]
    elif image.shape[-1] < shape[-1]:
        image = F.pad(image, (0, shape[-1] - image.shape[-1]))
    assert image.shape[-2:] == shape[-2:], f"{image.shape[-2:]} != {shape[-2:]}"
    return image


def check_nan(val: torch.Tensor, name: str) -> None:
    if torch.isnan(val).any():
        raise ValueError(f"NaN detected in {name}.")
