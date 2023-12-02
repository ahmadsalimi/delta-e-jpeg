import argparse
import os

import kornia as K
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize an image as channels',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', type=str, help='Path to the image')
    parser.add_argument('--cspace', type=str, default='rgb', help='Color space to use',
                        choices=['rgb', 'ycbcr'])
    args = parser.parse_args()

    image = K.io.load_image(args.path, K.io.ImageLoadType.RGB32)
    src_directory = os.path.dirname(args.path)
    basename, _ = os.path.splitext(os.path.basename(args.path))
    directory = os.path.join(src_directory, f'{basename}_{args.cspace}')
    os.makedirs(directory, exist_ok=True)

    if args.cspace == 'rgb':
        r, g, b = torch.unbind(image, dim=0)

        red_only = torch.stack([r, torch.zeros_like(g), torch.zeros_like(b)], dim=0)
        green_only = torch.stack([torch.zeros_like(r), g, torch.zeros_like(b)], dim=0)
        blue_only = torch.stack([torch.zeros_like(r), torch.zeros_like(g), b], dim=0)

        plt.imsave(os.path.join(directory, 'r.png'), red_only.permute(1, 2, 0).cpu().numpy())
        plt.imsave(os.path.join(directory, 'g.png'), green_only.permute(1, 2, 0).cpu().numpy())
        plt.imsave(os.path.join(directory, 'b.png'), blue_only.permute(1, 2, 0).cpu().numpy())

    elif args.cspace == 'ycbcr':
        ycbcr = K.color.rgb_to_ycbcr(image)
        y, cb, cr = torch.unbind(ycbcr, dim=0)

        y_only = y.expand(3, -1, -1).clamp(0, 1)
        half = torch.ones_like(y) * 0.5
        cb_only = K.color.ycbcr_to_rgb(torch.stack([half, cb, half], dim=0)).clamp(0, 1)
        cr_only = K.color.ycbcr_to_rgb(torch.stack([half, half, cr], dim=0)).clamp(0, 1)

        plt.imsave(os.path.join(directory, 'y.png'), y_only.permute(1, 2, 0).cpu().numpy())
        plt.imsave(os.path.join(directory, 'cb.png'), cb_only.permute(1, 2, 0).cpu().numpy())
        plt.imsave(os.path.join(directory, 'cr.png'), cr_only.permute(1, 2, 0).cpu().numpy())

    else:
        raise ValueError(f'Unknown color space {args.cspace}')
