import torch
from torch import nn


class Swish(nn.Module):
    """
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py
    ### Swish actiavation function

    $$x \cdot \sigma(x)$$
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
