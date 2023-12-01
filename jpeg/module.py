from typing import Tuple, Optional, Dict, Any, Literal

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F

from jpeg.jpeg import ExtendedJPEG
from jpeg.ds.conv import ConvDownsample, ConvUpsample
from metric.delta_e import DeltaE76, DeltaE2000
from metric.lpips import LPIPS
from metric.mae import MAE
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
                 optimizer: str = 'adam',
                 lp_kernel_size: int = 11,
                 lp_sigma: int = 3,
                 predict_model: Literal['jpeg', 'ejpeg'] = 'ejpeg'):
        super().__init__()
        downsample = downsample or ConvDownsample(64)
        upsample = upsample or ConvUpsample(64)
        self.ejpeg = ExtendedJPEG(downsample=downsample, upsample=upsample, quality=quality,
                                  lp_kernel_size=lp_kernel_size, lp_sigma=lp_sigma)
        self.jpeg = ExtendedJPEG(quality=quality, lp_kernel_size=lp_kernel_size, lp_sigma=lp_sigma)
        self.metrics = nn.ModuleDict(metrics or {
            'deltaE76': DeltaE76(),
            'deltaE2000': DeltaE2000(),
            'mse': MSE(),
            'mae': MAE(),
            'lpips_alex': LPIPS(net='alex'),
            'psnr': PSNR(data_range=1),
            'sparsity': Sparsity(),
        })
        self.loss_dict = loss_dict or {
            'mae': 1,
        }
        self.hparams.update(lr=lr,
                            weight_decay=weight_decay,
                            quality=quality,
                            optimizer=optimizer,
                            predict_model=predict_model)

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
    def full_forward(model: ExtendedJPEG, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb = F.pad(F.pad(rgb, (16, 16, 16, 16), mode='reflect'), (16, 16, 16, 16))
        y, cbcr = model.encode(rgb)
        rgb_hat = model.decode(y, cbcr, rgb.shape[-2:])
        rgb_hat = rgb_hat[..., 32:-32, 32:-32]
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
        loss = sum(self.metrics[name](x=x, x_hat=x_hat) * weight
                   for name, weight in self.loss_dict.items())
        self.log('train_loss', loss)
        with torch.no_grad():
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

    @torch.no_grad()
    def predict_step(self, x: torch.Tensor, batch_idx: int,
                     dataloader_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self.ejpeg if self.hparams['predict_model'] == 'ejpeg' else self.jpeg
        x = x.to(self.device)
        x = F.pad(F.pad(x, (16, 16, 16, 16), mode='reflect'), (16, 16, 16, 16))
        y, cbcr = model.get_mapping(x)
        x_hat = model.decode(y, cbcr, x.shape[-2:])
        y = y[..., 32:-32, 32:-32]
        cbcr = cbcr[..., 16:-16, 16:-16]
        x_hat = x_hat[..., 32:-32, 32:-32]
        return x_hat, y, cbcr

