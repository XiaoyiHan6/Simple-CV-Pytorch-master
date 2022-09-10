import torch
from torch import nn
from utils.L2Norm import L2Norm
from torch.cuda.amp import autocast
from torchvision import models


# "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
# "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",

def add_vgg():
    vgg16 = models.vgg16()
    vggs = vgg16.features
    vggs[16] = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
    vggs[-1] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
    conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)
    relu = nn.ReLU(inplace=True)
    vggs.add_module('31', conv6)
    vggs.add_module('32', relu)
    vggs.add_module('33', conv7)
    vggs.add_module('34', relu)
    return vggs


def add_extras(batch_norm=False):
    """
           batch_norm: whether to use BN
    """
    layers = []
    conv8_1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1)
    batch_norm8_1 = nn.BatchNorm2d(256)

    conv8_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
    batch_norm8_2 = nn.BatchNorm2d(512)

    conv9_1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1)
    batch_norm9_1 = nn.BatchNorm2d(128)

    conv9_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
    batch_norm9_2 = nn.BatchNorm2d(256)

    conv10_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)
    batch_norm10_1 = nn.BatchNorm2d(128)

    conv10_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
    batch_norm10_2 = nn.BatchNorm2d(256)

    conv11_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)
    batch_norm11_1 = nn.BatchNorm2d(128)

    conv11_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
    batch_norm11_2 = nn.BatchNorm2d(256)

    relu = nn.ReLU(inplace=True)

    if batch_norm:
        layers = [conv8_1, batch_norm8_1, relu,
                  conv8_2, batch_norm8_2, relu,
                  conv9_1, batch_norm9_1, relu,
                  conv9_2, batch_norm9_2, relu,
                  conv10_1, batch_norm10_1, relu,
                  conv10_2, batch_norm10_2, relu,
                  conv11_1, batch_norm11_1, relu,
                  conv11_2, batch_norm11_2, relu]
    else:
        layers = [conv8_1, relu,
                  conv8_2, relu,
                  conv9_1, relu,
                  conv9_2, relu,
                  conv10_1, relu,
                  conv10_2, relu,
                  conv11_1, relu,
                  conv11_2, relu]
    return layers


class VggNetBackbone(nn.Module):
    def __init__(self, batch_norm=False):
        super(VggNetBackbone, self).__init__()
        self.vgg = add_vgg()
        self.extras = nn.ModuleList(add_extras(batch_norm=batch_norm))
        self.l2_norm = L2Norm(512, scale=20)
        self.batch_norm = batch_norm

    @autocast()
    def forward(self, x):
        features = []
        for i in range(23):
            x = self.vgg[i](x)  # Conv4_3
        s = self.l2_norm(x)  # L2 normalization
        features.append(s)

        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)  # Conv 7
        features.append(x)

        for k, v in enumerate(self.extras):
            x = v(x)
            if self.batch_norm:
                if k % 6 == 5:
                    features.append(x)
            else:
                if k % 4 == 3:
                    features.append(x)
        return features


if __name__ == "__main__":
    layers = VggNetBackbone(batch_norm=False)
    x = torch.randn(16, 3, 300, 300)
    layers(x)
