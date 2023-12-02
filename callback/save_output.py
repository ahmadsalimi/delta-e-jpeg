import os
from typing import Any, Tuple, Sequence

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from data.dataset import ImageFolder


class SaveOutput(pl.callbacks.BasePredictionWriter):

    def __init__(self, directory: str) -> None:
        super().__init__(write_interval='batch')
        self.directory = directory
        self.filenames = None
        self.x = None
        self.x_hat = None
        self.y = None
        self.cbcr = None

    def on_predict_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.filenames = []
        self.x = torch.tensor([], device=pl_module.device)
        self.x_hat = torch.tensor([], device=pl_module.device)
        self.y = torch.tensor([], device=pl_module.device)
        self.cbcr = torch.tensor([], device=pl_module.device)

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        x_hat, y, cbcr = outputs
        dataloader = trainer.predict_dataloaders[dataloader_idx]
        dataset: ImageFolder = dataloader.dataset
        batch_start = batch_idx * trainer.datamodule.batch_size
        batch_files = dataset.files[batch_start:batch_start + len(x_hat)]
        self.filenames += list(map(os.path.basename, batch_files))
        self.x = torch.cat((self.x, batch), dim=0)
        self.x_hat = torch.cat((self.x_hat, x_hat), dim=0)
        self.y = torch.cat((self.y, y), dim=0)
        self.cbcr = torch.cat((self.cbcr, cbcr), dim=0)

    def on_predict_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Sequence[Any]
    ) -> None:
        directory = os.path.join(trainer.default_root_dir, self.directory)
        os.makedirs(directory, exist_ok=True)
        assert len(self.filenames) == len(self.x) == len(self.x_hat) == len(self.y) == len(self.cbcr)
        for filename, x, x_hat, y, cbcr in zip(self.filenames, self.x, self.x_hat, self.y, self.cbcr):
            prefix, ext = os.path.splitext(filename)
            plt.imsave(os.path.join(directory, f'{prefix}{ext}'), x.permute(1, 2, 0).cpu().numpy())
            plt.imsave(os.path.join(directory, f'{prefix}_hat{ext}'), x_hat.permute(1, 2, 0).cpu().numpy())
            plt.imsave(os.path.join(directory, f'{prefix}_y{ext}'), y.squeeze(0).cpu().numpy(), cmap='gray')
            plt.imsave(os.path.join(directory, f'{prefix}_cbcr0{ext}'), cbcr[0].cpu().numpy(), cmap='gray')
            plt.imsave(os.path.join(directory, f'{prefix}_cbcr1{ext}'), cbcr[1].cpu().numpy(), cmap='gray')
