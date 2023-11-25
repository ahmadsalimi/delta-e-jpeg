import os

from pytorch_lightning import LightningDataModule
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader


class DataModule(LightningDataModule):

    def __init__(self, data_dir: str,
                 batch_size: int = 512,
                 num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def setup(self, stage: str) -> None:
        self.train_dataset = ImageFolder(os.path.join(self.data_dir, 'train'),
                                         transform=self.transform)
        self.val_dataset = ImageFolder(os.path.join(self.data_dir, 'val'),
                                       transform=self.transform)
        self.test_dataset = ImageFolder(os.path.join(self.data_dir, 'test'),
                                        transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
