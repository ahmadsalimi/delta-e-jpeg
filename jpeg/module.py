from typing import Tuple, Optional, Dict

import torch
from pytorch_lightning import LightningModule
from torch import nn

from jpeg.jpeg import ExtendedJPEG
from jpeg.ds.conv import ConvDownsample, ConvUpsample
from metric.delta_e import DeltaE76, DeltaE2000
from metric.lpips import LPIPS
from metric.mae import MAE, LowPassMAE
from metric.mse import MSE
from metric.psnr import PSNR
from metric.sparsity import Sparsity


class ExtendedJPEGModule(LightningModule):
    def __init__(self, downsample: Optional[nn.Module] = None,
                 upsample: Optional[nn.Module] = None,
                 quality: int = 50,
                 metrics: Optional[Dict[str, nn.Module]] = None,
                 loss_dict: Optional[Dict[str, float]] = None,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-6,
                 optimizer: str = 'adam'):
        super().__init__()
        downsample = downsample or ConvDownsample(64)
        upsample = upsample or ConvUpsample(64)
        self.ejpeg = ExtendedJPEG(downsample=downsample, upsample=upsample, quality=quality)
        self.jpeg = ExtendedJPEG(quality=quality)
        self.metrics = nn.ModuleDict(metrics or {
            'deltaE76': DeltaE76(),
            'deltaE2000': DeltaE2000(),
            'mse': MSE(),
            'mae': MAE(),
            'lpips_alex': LPIPS(net='alex'),
            'lp_mae': LowPassMAE()
            # 'psnr': PSNR(data_range=1),
            # 'sparsity': Sparsity(),
        })
        self.loss_dict = loss_dict or {
            'mae': 1,
        }
        self.hparams.update(lr=lr,
                            weight_decay=weight_decay,
                            quality=quality,
                            optimizer=optimizer)

    # ignore metrics parameters in loading and saving the state dict
    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        sd = {
            key: value
            for key, value in sd.items()
            if not key.startswith('metrics.')
        }
        return sd

    def load_state_dict(self, state_dict, *args, **kwargs):
        state_dict = {**state_dict, **{
            key: value
            for key, value in super().state_dict().items()
            if key not in state_dict
        }}
        super().load_state_dict(state_dict, *args, **kwargs)

    def forward(self, rgb: torch.Tensor) -> Tuple[torch.Tensor]:
        return self.ejpeg(rgb)

    @staticmethod
    def full_forward(model: ExtendedJPEG, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y, cbcr = model.encode(rgb)
        rgb_hp, rgb_lp = model.decode(y, cbcr, rgb.shape[-2:])
        rgb_hat = rgb_hp + rgb_lp
        return y, cbcr, rgb_hat, rgb_hp, rgb_lp

    def __step_model(self, model: ExtendedJPEG, x: torch.Tensor, stage: str):
        y, cbcr, x_hat, x_hp, x_lp = self.full_forward(model, x)
        metrics = {
            name: metric(x=x, x_hat=x_hat, y=y, cbcr=cbcr, x_hp=x_hp, x_lp=x_lp)
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
        y, cbcr, x_hat, x_hp, x_lp = self.full_forward(self.ejpeg, x)
        loss = sum(self.metrics[name](x=x, x_hat=x_hat, y=y, cbcr=cbcr, x_hp=x_hp) * weight
                   for name, weight in self.loss_dict.items())
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
        if self.hparams['optimizer'] == 'adam':
            return torch.optim.Adam(self.ejpeg.parameters(),
                                    lr=self.hparams['lr'],
                                    weight_decay=self.hparams['weight_decay'])
        if self.hparams['optimizer'] == 'sgd':
            return torch.optim.SGD(self.ejpeg.parameters(),
                                   lr=self.hparams['lr'],
                                   weight_decay=self.hparams['weight_decay'])
