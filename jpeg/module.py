from typing import Tuple, Optional, Dict, Any

import torch
from pytorch_lightning import LightningModule
from torch import nn

from jpeg.jpeg import ExtendedJPEG
from jpeg.ds.conv import ConvDownsample, ConvUpsample
from metric.delta_e import DeltaE2000
from metric.mse import MSE
from metric.psnr import PSNR
from metric.sparsity import Sparsity


class ExtendedJPEGModule(LightningModule):
    def __init__(self, downsample: Optional[nn.Module] = None,
                 upsample: Optional[nn.Module] = None,
                 metrics: Optional[Dict[str, nn.Module]] = None,
                 loss_dict: Optional[Dict[str, float]] = None,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-6):
        super().__init__()
        downsample = downsample or ConvDownsample(64)
        upsample = upsample or ConvUpsample(64)
        self.ejpeg = ExtendedJPEG(downsample=downsample, upsample=upsample)
        self.jpeg = ExtendedJPEG()
        self.metrics = metrics or {
            'deltaE': DeltaE2000(),
            'mse': MSE(),
            'psnr': PSNR(data_range=1),
            'sparsity': Sparsity(),
        }
        self.loss_dict = loss_dict or {
            'deltaE': 1,
        }
        self.hparams.update(lr=lr,
                            weight_decay=weight_decay)

    def forward(self, rgb: torch.Tensor) -> Tuple[torch.Tensor]:
        return self.ejpeg(rgb)

    @staticmethod
    def full_forward(model: ExtendedJPEG, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y, cbcr = model.encode(rgb)
        rgb_hat = model.decode(y, cbcr, rgb.shape[-2:])
        return y, cbcr, rgb_hat

    def __step_model(self, model: ExtendedJPEG, x: torch.Tensor, stage: str):
        y, cbcr, x_hat = self.full_forward(model, x)
        metrics = {
            name: metric(x=x, x_hat=x_hat, y=y, cbcr=cbcr)
            for name, metric in self.metrics.items()
        }
        for name, metric in metrics.items():
            self.log(f'{stage}_{name}', metric, prog_bar=True)

    def __step(self, x: torch.Tensor, stage: str):
        self.__step_model(self.ejpeg, x, stage)
        self.__step_model(self.jpeg, x, f'{stage}_jpeg')

    def training_step(self, x: torch.Tensor, batch_idx: int) -> torch.Tensor:
        self.log('lr', self.optimizers().param_groups[0]['lr'], prog_bar=True)
        x = x.to(self.device)
        x_hat = self(x)
        loss = self.metrics['deltaE'](x=x, x_hat=x_hat) * self.loss_dict['deltaE']
        self.log('train_loss', loss)
        self.__step(x, 'train')
        return loss

    @torch.no_grad()
    def validation_step(self, x: torch.Tensor, batch_idx: int):
        self.__step(x.to(self.device), 'val')

    @torch.no_grad()
    def test_step(self, x: torch.Tensor, batch_idx: int):
        self.__step(x.to(self.device), 'test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.ejpeg.parameters(),
                                lr=self.hparams['lr'],
                                weight_decay=self.hparams['weight_decay'])
