from typing import Tuple

import torch
from torch import nn
import kornia as K


class DeltaE76(nn.Module):
    """Compute the delta E 76 between two batches of images."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the delta E 76 between two batches of images.

        Args:
            x (torch.Tensor): A batch of images in RGB format with shape :math:`(B, 3, H, W)`.
                The images are assumed to be in the range :math:`[0, 1]`.
            y (torch.Tensor): A batch of images in RGB format with shape :math:`(B, 3, H, W)`.
                The images are assumed to be in the range :math:`[0, 1]`.

        Returns:
            torch.Tensor: The delta E between the two batches of images with shape :math:`(B)`.
        """
        x = K.color.rgb_to_lab(x)
        y = K.color.rgb_to_lab(y)
        return (x - y).norm(dim=1).mean()


class DeltaE2000(nn.Module):
    """Compute the delta E 2000 between two batches of images.

    Args:
        k_l (float): The weight of the lightness. Defaults to 1.
        k_c (float): The weight of the chroma. Defaults to 1.
        k_h (float): The weight of the hue. Defaults to 1.
    """

    def __init__(self, k_l: float = 1, k_c: float = 1, k_h: float = 1):
        super().__init__()
        self.k_l = k_l
        self.k_c = k_c
        self.k_h = k_h

    @staticmethod
    def __c(x: torch.Tensor) -> torch.Tensor:
        return x[:, 1:].norm(dim=1)

    @staticmethod
    def __lab(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return K.color.rgb_to_lab(x).unbind(dim=1)

    @staticmethod
    def __bar(*x: torch.Tensor) -> torch.Tensor:
        return sum(x) / len(x)

    @staticmethod
    def __a_prime(a: torch.Tensor, c_bar: torch.Tensor) -> torch.Tensor:
        return a + a / 2 * (1 - (c_bar ** 7 / (c_bar ** 7 + 25 ** 7)).sqrt())

    @staticmethod
    def __c_prime(a_prime: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a_prime ** 2 + b ** 2).sqrt()

    @staticmethod
    def __h(a_prime: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.rad2deg(torch.atan2(b, a_prime))

    @staticmethod
    def __delta_h(h_x: torch.Tensor, h_y: torch.Tensor) -> torch.Tensor:
        return torch.where((h_x - h_y).abs() <= 180, h_y - h_x,
                           torch.where(h_y <= h_x, h_y - h_x + 360,
                                       h_y - h_x - 360))

    @staticmethod
    def __h_bar(h_x: torch.Tensor, h_y: torch.Tensor) -> torch.Tensor:
        return torch.where((h_x - h_y).abs() > 180, (h_x + h_y + 360) / 2,
                           (h_x + h_y) / 2)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the delta E 2000 between two batches of images.

        Args:
            x (torch.Tensor): A batch of images in RGB format with shape :math:`(B, 3, H, W)`.
                The images are assumed to be in the range :math:`[0, 1]`.
            y (torch.Tensor): A batch of images in RGB format with shape :math:`(B, 3, H, W)`.
                The images are assumed to be in the range :math:`[0, 1]`.

        Returns:
            torch.Tensor: The delta E between the two batches of images with shape :math:`(B)`.
        """
        l_x, a_x, b_x = self.__lab(x)                       # B x H x W
        l_y, a_y, b_y = self.__lab(y)                       # B x H x W
        delta_l = l_y - l_x                                 # B x H x W
        l_bar = self.__bar(l_x, l_y)                    # B x H x W
        c_x = self.__c(x)                                   # B x H x W
        c_y = self.__c(y)                                   # B x H x W
        c_bar = self.__bar(c_x, c_y)                    # B x H x W
        a_x_prime = self.__a_prime(a_x, c_bar)              # B x H x W
        a_y_prime = self.__a_prime(a_y, c_bar)              # B x H x W
        c_x_prime = self.__c_prime(a_x_prime, b_x)          # B x H x W
        c_y_prime = self.__c_prime(a_y_prime, b_y)          # B x H x W
        delta_c_prime = c_y_prime - c_x_prime               # B x H x W
        c_bar_prime = self.__bar(c_x_prime, c_y_prime)  # B x H x W
        h_x = self.__h(a_x_prime, b_x)                      # B x H x W
        h_y = self.__h(a_y_prime, b_y)                      # B x H x W
        delta_h = self.__delta_h(h_x, h_y)                  # B x H x W
        delta_H = 2 * (c_x_prime * c_y_prime).sqrt() * \
            torch.sin(torch.deg2rad(delta_h) / 2)           # B x H x W
        h_bar = self.__h_bar(h_x, h_y)                      # B x H x W
        t = 1 - 0.17 * torch.cos(torch.deg2rad(h_bar - 30)) + \
            0.24 * torch.cos(torch.deg2rad(2 * h_bar)) + \
            0.32 * torch.cos(torch.deg2rad(3 * h_bar + 6)) - \
            0.2 * torch.cos(torch.deg2rad(4 * h_bar - 63))  # B x H x W
        s_l = 1 + 0.015 * (l_bar - 50) ** 2 / \
            (20 + (l_bar - 50) ** 2).sqrt()                 # B x H x W
        s_c = 1 + 0.045 * c_bar_prime                       # B x H x W
        s_h = 1 + 0.015 * c_bar_prime * t                   # B x H x W
        r_t = -2 * (c_bar_prime ** 7 / (c_bar_prime ** 7 + 25 ** 7)).sqrt() * \
            torch.sin(
                torch.deg2rad(
                    60 * torch.exp(
                        -((h_bar - 275) / 25) ** 2)
                )
            )                                               # B x H x W
        delta_e = (delta_l / (self.k_l * s_l)) ** 2 + \
            (delta_c_prime / (self.k_c * s_c)) ** 2 + \
            (delta_H / (self.k_h * s_h)) ** 2 + \
            r_t * (delta_c_prime / (self.k_c * s_c)) * \
            (delta_H / (self.k_h * s_h))                    # B x H x W
        return delta_e.mean()
