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

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor, *_, **__) -> torch.Tensor:
        l_1, a_1, b_1 = self.__lab(x)                       # B x H x W
        l_2, a_2, b_2 = self.__lab(x_hat)                   # B x H x W

        avg_Lp = (l_1 + l_2) / 2

        C1 = torch.sqrt(a_1 ** 2 + b_1 ** 2)
        C2 = torch.sqrt(a_2 ** 2 + b_2 ** 2)

        avg_C1_C2 = (C1 + C2) / 2

        G = 0.5 * (1 - torch.sqrt(avg_C1_C2 ** 7 / (avg_C1_C2 ** 7 + 25 ** 7) + 1e-8))

        a1p = a_1 * (1 + G)
        a2p = a_2 * (1 + G)

        C1p = torch.sqrt(a1p ** 2 + b_1 ** 2)
        C2p = torch.sqrt(a2p ** 2 + b_2 ** 2)

        avg_C1p_C2p = (C1p + C2p) / 2

        h1p = torch.rad2deg(torch.atan2(b_1, a1p))
        h1p += (h1p < 0) * 360

        h2p = torch.rad2deg(torch.atan2(b_2, a2p))
        h2p += (h2p < 0) * 360

        avg_Hp = (((torch.abs(h1p - h2p) > 180) * 360) + h1p + h2p) / 2

        T = 1 - 0.17 * torch.cos(torch.deg2rad(avg_Hp - 30)) + \
            0.24 * torch.cos(torch.deg2rad(2 * avg_Hp)) + \
            0.32 * torch.cos(torch.deg2rad(3 * avg_Hp + 6)) - \
            0.2 * torch.cos(torch.deg2rad(4 * avg_Hp - 63))

        diff_h2p_h1p = h2p - h1p
        delta_hp = diff_h2p_h1p + (torch.abs(diff_h2p_h1p) > 180) * 360
        delta_hp -= (h2p > h1p) * 720

        delta_Lp = l_2 - l_1
        delta_Cp = C2p - C1p
        delta_Hp = 2 * torch.sqrt(C1p * C2p) * torch.sin(torch.deg2rad(delta_hp) / 2)

        S_L = 1 + ((0.015 * (avg_Lp - 50) ** 2) / torch.sqrt(20 + (avg_Lp - 50) ** 2))
        S_C = 1 + 0.045 * avg_C1p_C2p
        S_H = 1 + 0.015 * avg_C1p_C2p * T

        delta_ro = 30 * torch.exp(-(((avg_Hp - 275) / 25) ** 2))
        R_C = torch.sqrt((avg_C1p_C2p ** 7) / (avg_C1p_C2p ** 7 + 25 ** 7))
        R_T = -2 * R_C * torch.sin(2 * torch.deg2rad(delta_ro))

        delta_E = torch.sqrt(
            (delta_Lp / (self.k_l * S_L)) ** 2 +
            (delta_Cp / (self.k_c * S_C)) ** 2 +
            (delta_Hp / (self.k_h * S_H)) ** 2 +
            R_T * (delta_Cp / (self.k_c * S_C)) * (delta_Hp / (self.k_h * S_H))
        )
        return delta_E.mean()

