import glob
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageFolder(Dataset):

    def __init__(self, root: str, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.files = glob.glob(os.path.join(root, '*.png'))

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.files[index])
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self) -> int:
        return len(self.files)
