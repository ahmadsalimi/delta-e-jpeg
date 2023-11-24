from pytorch_lightning.cli import LightningCLI
from jpeg.module import ExtendedJPEGModule


def cli_main():
    cli = LightningCLI(ExtendedJPEGModule), #JPEGDataModule)
    # note: don't call fit!!


if __name__ == '__main__':
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
