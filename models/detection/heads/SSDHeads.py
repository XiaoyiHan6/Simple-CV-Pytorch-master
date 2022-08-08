import torch
from torch import nn
from torch.cuda.amp import autocast
from models.detection.necks.SSDNecks import SSDNecks
from models.detection.backbones.VggNetBackbone import VggNetBackbone


class SSDHeads(nn.Module):
    def __init__(self, vgg, neck, cfg, num_classes, batch_norm=False):
        super(SSDHeads, self).__init__()
        """
        cfg: channels of layer
        i: nput_channels
        batch_norm: whether to use BN
        """
        loc_layers = []
        conf_layers = []
        if batch_norm:
            vgg_source = [30, -3]
            neck_source = neck[3::6]
        else:
            vgg_source = [21, -2]
            neck_source = neck[2::4]
        for k, v in enumerate(vgg_source):
            loc_layers += [
                nn.Conv2d(in_channels=vgg[v].out_channels, out_channels=cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [
                nn.Conv2d(in_channels=vgg[v].out_channels, out_channels=cfg[k] * num_classes, kernel_size=3,
                          padding=1)]
        for k, v in enumerate(neck_source, 2):
            loc_layers += [
                nn.Conv2d(in_channels=v.out_channels, out_channels=cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [
                nn.Conv2d(in_channels=v.out_channels, out_channels=cfg[k] * num_classes, kernel_size=3, padding=1)]

        self.loc_head = nn.Sequential(*loc_layers)
        self.conf_head = nn.Sequential(*conf_layers)

    @autocast()
    def forward(self, x):
        loc_head = self.loc_head(x)
        conf_head = self.conf_head(x)

        return loc_head, conf_head


class SSDHeads_modify(nn.Module):
    def __init__(self, vgg, neck, cfg, num_classes, batch_norm=False):
        super(SSDHeads_modify, self).__init__()
        if batch_norm:
            loc1 = nn.Conv2d(in_channels=vgg[30].out_channels, out_channels=4 * 4, kernel_size=3, stride=1, padding=1)
            loc2 = nn.Conv2d(in_channels=vgg[-3].out_channels, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)
            loc3 = nn.Conv2d(in_channels=neck[3].out_channels, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)
            loc4 = nn.Conv2d(in_channels=neck[9].out_channels, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)
            loc5 = nn.Conv2d(in_channels=neck[15].out_channels, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)
            loc6 = nn.Conv2d(in_channels=neck[21].out_channels, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)

            conf1 = nn.Conv2d(in_channels=vgg[30].out_channels, out_channels=4 * num_classes, kernel_size=3, stride=1,
                              padding=1)
            conf2 = nn.Conv2d(in_channels=vgg[-3].out_channels, out_channels=6 * num_classes, kernel_size=3, stride=1,
                              padding=1)
            conf3 = nn.Conv2d(in_channels=neck[3].out_channels, out_channels=6 * num_classes, kernel_size=3, stride=1,
                              padding=1)
            conf4 = nn.Conv2d(in_channels=neck[9].out_channels, out_channels=6 * num_classes, kernel_size=3, stride=1,
                              padding=1)
            conf5 = nn.Conv2d(in_channels=neck[15].out_channels, out_channels=4 * num_classes, kernel_size=3, stride=1,
                              padding=1)
            conf6 = nn.Conv2d(in_channels=neck[21].out_channels, out_channels=4 * num_classes, kernel_size=3, stride=1,
                              padding=1)
        else:
            loc1 = nn.Conv2d(in_channels=vgg[21].out_channels, out_channels=4 * 4, kernel_size=3, stride=1, padding=1)
            loc2 = nn.Conv2d(in_channels=vgg[-2].out_channels, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)
            loc3 = nn.Conv2d(in_channels=neck[2].out_channels, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)
            loc4 = nn.Conv2d(in_channels=neck[6].out_channels, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)
            loc5 = nn.Conv2d(in_channels=neck[10].out_channels, out_channels=4 * 4, kernel_size=3, stride=1, padding=1)
            loc6 = nn.Conv2d(in_channels=neck[14].out_channels, out_channels=4 * 4, kernel_size=3, stride=1, padding=1)

            conf1 = nn.Conv2d(in_channels=vgg[21].out_channels, out_channels=4 * num_classes, kernel_size=3, stride=1,
                              padding=1)
            conf2 = nn.Conv2d(in_channels=vgg[-2].out_channels, out_channels=6 * num_classes, kernel_size=3, stride=1,
                              padding=1)
            conf3 = nn.Conv2d(in_channels=neck[2].out_channels, out_channels=6 * num_classes, kernel_size=3, stride=1,
                              padding=1)
            conf4 = nn.Conv2d(in_channels=neck[6].out_channels, out_channels=6 * num_classes, kernel_size=3, stride=1,
                              padding=1)
            conf5 = nn.Conv2d(in_channels=neck[10].out_channels, out_channels=4 * num_classes, kernel_size=3, stride=1,
                              padding=1)
            conf6 = nn.Conv2d(in_channels=neck[14].out_channels, out_channels=4 * num_classes, kernel_size=3, stride=1,
                              padding=1)

        loc_layers = [loc1, loc2, loc3, loc4, loc5, loc6]
        conf_layers = [conf1, conf2, conf3, conf4, conf5, conf6]
        self.loc_head = nn.Sequential(*loc_layers)
        self.conf_head = nn.Sequential(*conf_layers)

    def forward(self, x):
        loc_head = self.loc_head(x)
        conf_head = self.conf_head(x)
        return loc_head, conf_head


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
    backbones_1 = VggNetBackbone(backbone['300'], 3)
    necks_1 = SSDNecks(neck['300'], 1024)
    layers_1 = SSDHeads(backbones_1.features, necks_1.neck, head['300'], 21)
    print(layers_1)
    backbones_2 = VggNetBackbone(backbone['300'], 3, batch_norm=True)
    necks_2 = SSDNecks(neck['300'], 1024, batch_norm=True)
    layers_2 = SSDHeads_modify(backbones_2.features, necks_2.neck, head['300'], 21, batch_norm=True)
    print(layers_2)
