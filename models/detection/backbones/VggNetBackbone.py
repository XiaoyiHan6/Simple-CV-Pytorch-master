import torch
from torch import nn
from torch.cuda.amp import autocast


class VggNetBackbone(nn.Module):
    def __init__(self, cfg, i, batch_norm=False):
        super(VggNetBackbone, self).__init__()
        """
        cfg: channels of layer
        i: nput_channels
        batch_norm: whether to use BN
        """
        layers = []
        in_channels = i
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif v == 'C':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
            else:
                conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
                else:
                    layers += [conv2d, nn.ReLU()]
                in_channels = v
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6)
        batch_norm6 = nn.BatchNorm2d(1024)
        conv7 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)
        batch_norm7 = nn.BatchNorm2d(1024)
        if batch_norm:
            layers += [pool5, conv6, batch_norm6, nn.ReLU(), conv7, batch_norm7, nn.ReLU()]
        else:
            layers += [pool5, conv6, nn.ReLU(), conv7, nn.ReLU()]
        self.features = nn.Sequential(*layers)

    @autocast()
    def forward(self, x):
        x = self.features(x)
        return x


if __name__ == "__main__":
    backbone = {
        '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
        '512': [],
    }
    layers_1 = VggNetBackbone(backbone['300'], 3, batch_norm=True)
    print(layers_1)
    layers_2 = VggNetBackbone(backbone['300'], 3)
    print(layers_2)
