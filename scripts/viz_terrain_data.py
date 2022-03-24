import argparse
import os
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='dir of data to viz')
    args = parser.parse_args()

    for basename in os.listdir(args.data_dir):
        fp = os.path.join(args.data_dir, basename)
        hmap = torch.load(fp)
        metadata = hmap['metadata']
        data = hmap['data']

        xmin = metadata['origin'][0]
        xmax = xmin + metadata['width']
        ymin = metadata['origin'][1]
        ymax = ymin + metadata['height']

        plt.imshow(data, origin='lower', extent=(xmin, xmax, ymin, ymax))

        plt.title(fp)
        plt.xlabel('X(m)')
        plt.ylabel('Y(m)')
        plt.colorbar()
        plt.show()
