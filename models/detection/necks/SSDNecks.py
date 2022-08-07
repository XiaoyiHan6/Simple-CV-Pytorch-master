import torch
from torch import nn


def SSDExtraLayers(cfg, i, batch_norm=False):
    """
    cfg: channels of layer
    i: nput_channels
    batch_norm: whether to use BN
    """
    layers = []
    # 1024
    in_channels = i
    # flag choose kernel_size = 1 or 3
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=cfg[k + 1],
                                        kernel_size=(1, 3)[flag], stride=2, padding=1))
            else:
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=(1, 3)[flag]))
            flag = not flag
        in_channels = v
    return layers


if __name__ == "__main__":
    extras = {
        '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
        '512': [],
    }
    layers = SSDExtraLayers(extras['300'], 1024)
    extras = nn.Sequential(*layers)
    print(extras)
