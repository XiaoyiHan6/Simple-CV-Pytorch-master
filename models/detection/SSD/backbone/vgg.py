import torch
from torch import nn


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    if batch_norm:
        layers += [pool5, conv6, nn.BatchNorm2d(1024),
                   nn.ReLU(inplace=True), conv7, nn.BatchNorm2d(1024), nn.ReLU(inplace=True)]
    else:
        layers += [pool5, conv6,
                   nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(batch_norm=False):
    # Extra layers added to VGG for feature scaling
    exts1_1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
    bn1_1 = nn.BatchNorm2d(256)
    exts1_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
    bn1_2 = nn.BatchNorm2d(512)
    exts2_1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0)
    bn2_1 = nn.BatchNorm2d(128)
    exts2_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
    bn2_2 = nn.BatchNorm2d(256)
    exts3_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
    bn3_1 = nn.BatchNorm2d(128)
    exts3_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
    bn3_2 = nn.BatchNorm2d(256)
    exts4_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
    bn4_1 = nn.BatchNorm2d(128)
    exts4_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
    bn4_2 = nn.BatchNorm2d(256)

    relu = nn.ReLU(inplace=True)
    if batch_norm:
        return [exts1_1, bn1_1, relu,
                exts1_2, bn1_2, relu,
                exts2_1, bn2_1, relu,
                exts2_2, bn2_2, relu,
                exts3_1, bn3_1, relu,
                exts3_2, bn3_2, relu,
                exts4_1, bn4_1, relu,
                exts4_2, bn4_2, relu]
    else:
        return [exts1_1, relu,
                exts1_2, relu,
                exts2_1, relu,
                exts2_2, relu,
                exts3_1, relu,
                exts3_2, relu,
                exts4_1, relu,
                exts4_2, relu]


if __name__ == '__main__':
    base = {
        '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                512, 512, 512],
        '512': [],
    }
    extras = {
        '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
        '512': [],
    }
    vgg = nn.Sequential(*vgg(cfg=base['300'], i=3, batch_norm=True))
    print("vgg:", vgg)
    img = torch.randn(16, 3, 300, 300)
    output = vgg(img)
    print(output.shape)

    layers = nn.Sequential(*add_extras(batch_norm=True))
    print("layers:", layers)
