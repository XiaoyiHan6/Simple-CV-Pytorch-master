import torch
from torch import nn
from torch.cuda.amp import autocast
from models.detection.necks.SSDNecks import SSDNecks
from models.detection.backbones.VggNetBackbone import VggNetBackbone


class confHeads(nn.Module):
    def __init__(self, vgg, neck, cfg, num_classes, batch_norm=False):
        super(confHeads, self).__init__()
        self.num_classes = num_classes
        v = 0
        conf1 = nn.Conv2d(in_channels=vgg.block4_3[0].out_channels, out_channels=cfg[v] * num_classes, kernel_size=3,
                          stride=1,
                          padding=1)

        v += 1
        conf2 = nn.Conv2d(in_channels=vgg.block7[0].out_channels, out_channels=cfg[v] * num_classes, kernel_size=3,
                          stride=1,
                          padding=1)

        v += 1
        conf3 = nn.Conv2d(in_channels=neck.block8_2[0].out_channels, out_channels=cfg[v] * num_classes, kernel_size=3,
                          stride=1,
                          padding=1)

        v += 1
        conf4 = nn.Conv2d(in_channels=neck.block9_2[0].out_channels, out_channels=cfg[v] * num_classes, kernel_size=3,
                          stride=1,
                          padding=1)

        v += 1
        conf5 = nn.Conv2d(in_channels=neck.block10_2[0].out_channels, out_channels=cfg[v] * num_classes, kernel_size=3,
                          stride=1,
                          padding=1)

        v += 1
        conf6 = nn.Conv2d(in_channels=neck.block11_2[0].out_channels, out_channels=cfg[v] * num_classes, kernel_size=3,
                          stride=1,
                          padding=1)
        relu = nn.ReLU()
        sigmoid = nn.Sigmoid()
        self.conf_head = nn.Sequential(conf1, relu, conf2, relu, conf3, relu, conf4, relu, conf5, relu, conf6, sigmoid)

    @autocast()
    def forward(self, x):
        x = self.conf_head(x)
        out = x.permute(0, 2, 3, 1)
        out = out.contiguous().view(out.shape[0], -1, self.num_classes)
        del x
        return out


class locHeads(nn.Module):
    def __init__(self, vgg, neck, cfg, batch_norm=False):
        super(locHeads, self).__init__()
        v = 0
        loc1 = nn.Conv2d(in_channels=vgg.block4_3[0].out_channels, out_channels=cfg[v] * 4, kernel_size=3, stride=1,
                         padding=1)

        v += 1
        loc2 = nn.Conv2d(in_channels=vgg.block7[0].out_channels, out_channels=cfg[v] * 4, kernel_size=3, stride=1,
                         padding=1)

        v += 1
        loc3 = nn.Conv2d(in_channels=neck.block8_2[0].out_channels, out_channels=cfg[v] * 4, kernel_size=3, stride=1,
                         padding=1)

        v += 1
        loc4 = nn.Conv2d(in_channels=neck.block9_2[0].out_channels, out_channels=cfg[v] * 4, kernel_size=3, stride=1,
                         padding=1)

        v += 1
        loc5 = nn.Conv2d(in_channels=neck.block10_2[0].out_channels, out_channels=cfg[v] * 4, kernel_size=3, stride=1,
                         padding=1)

        v += 1
        loc6 = nn.Conv2d(in_channels=neck.block11_2[0].out_channels, out_channels=cfg[v] * 4, kernel_size=3, stride=1,
                         padding=1)
        relu = nn.ReLU()

        self.loc_head = nn.Sequential(loc1, relu, loc2, relu, loc3, relu, loc4, relu, loc5, relu, loc6)

    @autocast()
    def forward(self, x):
        x = self.loc_head(x)
        out = x.permute(0, 2, 3, 1)
        out = out.contiguous().view(out.shape[0], -1, 4)
        del x
        return out


if __name__ == "__main__":
    backbone = {
        '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
        '512': [],
    }
    neck = {
        '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
        '512': [],
    }
    head = {
        '300': [4, 6, 6, 6, 4, 4],
        '512': [],
    }
    backbones_1 = VggNetBackbone(backbone['300'], 3, batch_norm=True)
    necks_1 = SSDNecks(neck['300'], 1024, batch_norm=True)
    loc_head = locHeads(backbones_1, necks_1, head['300'], batch_norm=True)
    print(loc_head)

    backbones_2 = VggNetBackbone(backbone['300'], 3)
    necks_2 = SSDNecks(neck['300'], 1024)
    conf_head = confHeads(backbones_2, necks_2, head['300'], 21)
    print(conf_head)
