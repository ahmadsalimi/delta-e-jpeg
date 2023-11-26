import glob
import os

from torch.utils.data import Dataset
import kornia as K


class ImageFolder(Dataset):

    def __init__(self, root: str, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.files = glob.glob(os.path.join(root, '*.png'))

    def __getitem__(self, index: int):
        image = K.io.load_image(self.files[index], K.io.ImageLoadType.RGB32)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self) -> int:
        return len(self.files)
