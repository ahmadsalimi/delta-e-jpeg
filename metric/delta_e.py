from typing import Tuple

import torch
from torch import nn
import kornia as K

from jpeg.utils import check_nan


class DeltaE76(nn.Module):
    """Compute the delta E 76 between two batches of images."""

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor, *_, **__) -> torch.Tensor:
        """Compute the delta E 76 between two batches of images.

        Args:
            x (torch.Tensor): A batch of images in RGB format with shape :math:`(B, 3, H, W)`.
                The images are assumed to be in the range :math:`[0, 1]`.
            x_hat (torch.Tensor): A batch of images in RGB format with shape :math:`(B, 3, H, W)`.
                The images are assumed to be in the range :math:`[0, 1]`.

        Returns:
            torch.Tensor: The delta E between the two batches of images with shape :math:`(B)`.
        """
        x = K.color.rgb_to_lab(x)
        x_hat = K.color.rgb_to_lab(x_hat)
        return (x - x_hat).norm(dim=1).mean()


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
        return a + a / 2 * (1 - (c_bar ** 7 / (c_bar ** 7 + 25 ** 7) + 1e-8).sqrt())

    @staticmethod
    def __c_prime(a_prime: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a_prime ** 2 + b ** 2 + 1e-8).sqrt()

    @staticmethod
    def __h(a_prime: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.rad2deg(torch.atan2(b + 1e-8, a_prime))

    @staticmethod
    def __delta_h(h_x: torch.Tensor, h_y: torch.Tensor) -> torch.Tensor:
        return torch.where((h_x - h_y).abs() <= 180, h_y - h_x,
                           torch.where(h_y <= h_x, h_y - h_x + 360,
                                       h_y - h_x - 360))

    @staticmethod
    def __h_bar(h_x: torch.Tensor, h_y: torch.Tensor) -> torch.Tensor:
        return torch.where((h_x - h_y).abs() > 180, (h_x + h_y + 360) / 2,
                           (h_x + h_y) / 2)

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor, *_, **__) -> torch.Tensor:
        """Compute the delta E 2000 between two batches of images.

        Args:
            x (torch.Tensor): A batch of images in RGB format with shape :math:`(B, 3, H, W)`.
                The images are assumed to be in the range :math:`[0, 1]`.
            x_hat (torch.Tensor): A batch of images in RGB format with shape :math:`(B, 3, H, W)`.
                The images are assumed to be in the range :math:`[0, 1]`.

        Returns:
            torch.Tensor: The delta E between the two batches of images with shape :math:`(B)`.
        """
        l_x, a_x, b_x = self.__lab(x)                       # B x H x W
        check_nan(l_x, 'l_x')
        check_nan(a_x, 'a_x')
        check_nan(b_x, 'b_x')
        l_y, a_y, b_y = self.__lab(x_hat)                   # B x H x W
        try:
            check_nan(l_y, 'l_y')
            check_nan(a_y, 'a_y')
            check_nan(b_y, 'b_y')
        except:
            print(f'x_hat: {x_hat.min()}, {x_hat.max()}')
            raise
        delta_l = l_y - l_x                                 # B x H x W
        check_nan(delta_l, 'delta_l')
        l_bar = self.__bar(l_x, l_y)                    # B x H x W
        check_nan(l_bar, 'l_bar')
        c_x = self.__c(x)                                   # B x H x W
        check_nan(c_x, 'c_x')
        c_y = self.__c(x_hat)                               # B x H x W
        check_nan(c_y, 'c_y')
        c_bar = self.__bar(c_x, c_y)                    # B x H x W
        check_nan(c_bar, 'c_bar')
        a_x_prime = self.__a_prime(a_x, c_bar)              # B x H x W
        check_nan(a_x_prime, 'a_x_prime')
        a_y_prime = self.__a_prime(a_y, c_bar)              # B x H x W
        check_nan(a_y_prime, 'a_y_prime')
        c_x_prime = self.__c_prime(a_x_prime, b_x)          # B x H x W
        check_nan(c_x_prime, 'c_x_prime')
        c_y_prime = self.__c_prime(a_y_prime, b_y)          # B x H x W
        check_nan(c_y_prime, 'c_y_prime')
        delta_c_prime = c_y_prime - c_x_prime               # B x H x W
        check_nan(delta_c_prime, 'delta_c_prime')
        c_bar_prime = self.__bar(c_x_prime, c_y_prime)  # B x H x W
        check_nan(c_bar_prime, 'c_bar_prime')
        h_x = self.__h(a_x_prime, b_x)                      # B x H x W
        check_nan(h_x, 'h_x')
        h_y = self.__h(a_y_prime, b_y)                      # B x H x W
        check_nan(h_y, 'h_y')
        delta_h = self.__delta_h(h_x, h_y)                  # B x H x W
        check_nan(delta_h, 'delta_h')
        delta_H = 2 * (c_x_prime * c_y_prime + 1e-8).sqrt() * \
            torch.sin(torch.deg2rad(delta_h) / 2)           # B x H x W
        check_nan(delta_H, 'delta_H')
        h_bar = self.__h_bar(h_x, h_y)                      # B x H x W
        check_nan(h_bar, 'h_bar')
        t = 1 - 0.17 * torch.cos(torch.deg2rad(h_bar - 30)) + \
            0.24 * torch.cos(torch.deg2rad(2 * h_bar)) + \
            0.32 * torch.cos(torch.deg2rad(3 * h_bar + 6)) - \
            0.2 * torch.cos(torch.deg2rad(4 * h_bar - 63))  # B x H x W
        check_nan(t, 't')
        s_l = 1 + 0.015 * (l_bar - 50) ** 2 / \
            (20 + (l_bar - 50) ** 2 + 1e-8).sqrt()                 # B x H x W
        check_nan(s_l, 's_l')
        s_c = 1 + 0.045 * c_bar_prime                       # B x H x W
        check_nan(s_c, 's_c')
        s_h = 1 + 0.015 * c_bar_prime * t                   # B x H x W
        check_nan(s_h, 's_h')
        r_t = -2 * (c_bar_prime ** 7 / (c_bar_prime ** 7 + 25 ** 7) + 1e-8).sqrt() * \
            torch.sin(
                torch.deg2rad(
                    60 * torch.exp(
                        -((h_bar - 275) / 25) ** 2)
                )
            )                                               # B x H x W
        check_nan(r_t, 'r_t')
        delta_e = (delta_l / (self.k_l * s_l)) ** 2 + \
            (delta_c_prime / (self.k_c * s_c)) ** 2 + \
            (delta_H / (self.k_h * s_h)) ** 2 + \
            r_t * (delta_c_prime / (self.k_c * s_c)) * \
            (delta_H / (self.k_h * s_h))                    # B x H x W
        check_nan(delta_e, 'delta_e')
        return delta_e.mean()
