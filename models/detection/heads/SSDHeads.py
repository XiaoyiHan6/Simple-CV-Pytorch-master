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

        self.conf1 = nn.Sequential(conf1, relu)
        self.conf2 = nn.Sequential(conf2, relu)
        self.conf3 = nn.Sequential(conf3, relu)
        self.conf4 = nn.Sequential(conf4, relu)
        self.conf5 = nn.Sequential(conf5, relu)
        self.conf6 = nn.Sequential(conf6, relu)

    @autocast()
    def forward(self, x):
        out1 = self.conf1(x[0])
        out1 = out1.permute(0, 2, 3, 1).contiguous()

        out2 = self.conf2(x[1])
        out2 = out2.permute(0, 2, 3, 1).contiguous()

        out3 = self.conf3(x[2])
        out3 = out3.permute(0, 2, 3, 1).contiguous()

        out4 = self.conf4(x[3])
        out4 = out4.permute(0, 2, 3, 1).contiguous()

        out5 = self.conf5(x[4])
        out5 = out5.permute(0, 2, 3, 1).contiguous()

        out6 = self.conf6(x[5])
        out6 = out6.permute(0, 2, 3, 1).contiguous()

        del x
        out = [out1, out2, out3, out4, out5, out6]
        del out1, out2, out3, out4, out5, out6
        conf = torch.cat([o.view(o.size(0), -1) for o in out], dim=1)
        del out
        conf = conf.view(conf.size(0), -1, self.num_classes)
        return conf


class locHeads(nn.Module):
    def __init__(self, vgg, neck, cfg, batch_norm=False):
        super(locHeads, self).__init__()
        v = 0
        loc1 = nn.Conv2d(in_channels=vgg.block4_3[0].out_channels, out_channels=cfg[v] * 4, kernel_size=3,
                         stride=1,
                         padding=1)

        v += 1
        loc2 = nn.Conv2d(in_channels=vgg.block7[0].out_channels, out_channels=cfg[v] * 4, kernel_size=3,
                         stride=1,
                         padding=1)

        v += 1
        loc3 = nn.Conv2d(in_channels=neck.block8_2[0].out_channels, out_channels=cfg[v] * 4, kernel_size=3,
                         stride=1,
                         padding=1)

        v += 1
        loc4 = nn.Conv2d(in_channels=neck.block9_2[0].out_channels, out_channels=cfg[v] * 4, kernel_size=3,
                         stride=1,
                         padding=1)

        v += 1
        loc5 = nn.Conv2d(in_channels=neck.block10_2[0].out_channels, out_channels=cfg[v] * 4, kernel_size=3,
                         stride=1,
                         padding=1)

        v += 1
        loc6 = nn.Conv2d(in_channels=neck.block11_2[0].out_channels, out_channels=cfg[v] * 4, kernel_size=3,
                         stride=1,
                         padding=1)
        relu = nn.ReLU()

        self.loc1 = nn.Sequential(loc1, relu)
        self.loc2 = nn.Sequential(loc2, relu)
        self.loc3 = nn.Sequential(loc3, relu)
        self.loc4 = nn.Sequential(loc4, relu)
        self.loc5 = nn.Sequential(loc5, relu)
        self.loc6 = nn.Sequential(loc6, relu)

    @autocast()
    def forward(self, x):
        out1 = self.loc1(x[0])
        out1 = out1.permute(0, 2, 3, 1).contiguous()

        out2 = self.loc2(x[1])
        out2 = out2.permute(0, 2, 3, 1).contiguous()

        out3 = self.loc3(x[2])
        out3 = out3.permute(0, 2, 3, 1).contiguous()

        out4 = self.loc4(x[3])
        out4 = out4.permute(0, 2, 3, 1).contiguous()

        out5 = self.loc5(x[4])
        out5 = out5.permute(0, 2, 3, 1).contiguous()

        out6 = self.loc6(x[5])
        out6 = out6.permute(0, 2, 3, 1).contiguous()

        del x
        out = [out1, out2, out3, out4, out5, out6]
        del out1, out2, out3, out4, out5, out6
        loc = torch.cat([o.view(o.size(0), -1) for o in out], dim=1)
        del out
        loc = loc.view(loc.size(0), -1, 4)
        return loc


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
    conf_head = confHeads(backbones_2, necks_2, head['300'], 20)
    print(conf_head)
