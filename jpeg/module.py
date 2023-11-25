from typing import Tuple, Optional, Dict

import torch
from pytorch_lightning import LightningModule
from torch import nn

from jpeg.jpeg import ExtendedJPEG
from metric.delta_e import DeltaE2000
from metric.mse import MSE
from metric.psnr import PSNR


class ExtendedJPEGModule(LightningModule):
    def __init__(self, downsample: Optional[nn.Module] = None,
                 upsample: Optional[nn.Module] = None,
                 metrics: Optional[Dict[str, nn.Module]] = None,
                 main_loss: str = 'deltaE',
                 lr: float = 1e-3,
                 weight_decay: float = 1e-6):
        super().__init__()
        self.jpeg = ExtendedJPEG(downsample=downsample, upsample=upsample)
        self.metrics = metrics or {
            'deltaE': DeltaE2000(),
            'mse': MSE(),
            'psnr': PSNR(data_range=1),
        }
        self.main_loss = main_loss
        self.hparams.update(lr=lr,
                            weight_decay=weight_decay)

    def forward(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y, cbcr = self.jpeg.encode(rgb)
        rgb_hat = self.jpeg.decode(y, cbcr, rgb.shape[-2:])
        return y, cbcr, rgb_hat

    def training_step(self, x: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = x.to(self.device)
        y, cbcr, x_hat = self(x)
        metrics = {
            name: metric(x, x_hat)
            for name, metric in self.metrics.items()
        }
        self.log('train_loss', metrics[self.main_loss])
        for name, metric in metrics.items():
            self.log(f'train_{name}', metric)
        return metrics[self.main_loss]

    def validation_step(self, x: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = x.to(self.device)
        y, cbcr, x_hat = self(x)
        metrics = {
            name: metric(x, x_hat)
            for name, metric in self.metrics.items()
        }
        self.log('val_loss', metrics[self.main_loss])
        for name, metric in metrics.items():
            self.log(f'val_{name}', metric)
        return metrics[self.main_loss]

    def configure_optimizers(self):
        return torch.optim.Adam(self.epsilon_theta.parameters(),
                                lr=self.hparams['lr'],
                                weight_decay=self.hparams['weight_decay'])
