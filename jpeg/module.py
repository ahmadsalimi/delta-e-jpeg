from typing import Tuple, Optional

import torch
from pytorch_lightning import LightningModule
from torch import nn

from jpeg.jpeg import ExtendedJPEG
from metric.delta_e import DeltaE


class ExtendedJPEGModule(LightningModule):
    def __init__(self, downsample: Optional[nn.Module] = None,
                 upsample: Optional[nn.Module] = None,
                 loss: Optional[nn.Module] = None,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-6):
        super().__init__()
        self.jpeg = ExtendedJPEG(downsample=downsample, upsample=upsample)
        self.loss = loss or DeltaE()
        self.hparams.update(lr=lr,
                            weight_decay=weight_decay)

    def forward(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y, cbcr = self.jpeg.encode(rgb)
        rgb_hat = self.jpeg.decode(y, cbcr, rgb.shape[-2:])
        return y, cbcr, rgb_hat

    def training_step(self, x: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = x.to(self.device)
        y, cbcr, x_hat = self(x)
        loss = self.loss(x, x_hat)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, x: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = x.to(self.device)
        y, cbcr, x_hat = self(x)
        loss = self.loss(x, x_hat)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.epsilon_theta.parameters(),
                                lr=self.hparams['lr'],
                                weight_decay=self.hparams['weight_decay'])
