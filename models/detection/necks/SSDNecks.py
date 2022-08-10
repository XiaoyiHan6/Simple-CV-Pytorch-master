import torch
from torch import nn
from torch.cuda.amp import autocast


class SSDNecks(nn.Module):
    def __init__(self, cfg, i, batch_norm=False):
        super(SSDNecks, self).__init__()
        """
        cfg: channels of layer
        i: nput_channels
        batch_norm: whether to use BN
        """
        # 1024
        in_channels = i

        v = 0
        conv8_1 = nn.Conv2d(in_channels=in_channels, out_channels=cfg[v], kernel_size=1, stride=1)
        batch_norm8_1 = nn.BatchNorm2d(cfg[v])

        v += 2
        conv8_2 = nn.Conv2d(in_channels=cfg[v - 2], out_channels=cfg[v], kernel_size=3, stride=2, padding=1)
        batch_norm8_2 = nn.BatchNorm2d(cfg[v])

        v += 1
        conv9_1 = nn.Conv2d(in_channels=cfg[v - 1], out_channels=cfg[v], kernel_size=1, stride=1, padding=0)
        batch_norm9_1 = nn.BatchNorm2d(cfg[v])

        v += 2
        conv9_2 = nn.Conv2d(in_channels=cfg[v - 2], out_channels=cfg[v], kernel_size=3, stride=2, padding=1)
        batch_norm9_2 = nn.BatchNorm2d(cfg[v])

        v += 1
        conv10_1 = nn.Conv2d(in_channels=cfg[v - 1], out_channels=cfg[v], kernel_size=1, stride=1, padding=0)
        batch_norm10_1 = nn.BatchNorm2d(cfg[v])

        v += 1
        conv10_2 = nn.Conv2d(in_channels=cfg[v - 1], out_channels=cfg[v], kernel_size=3, stride=1, padding=0)
        batch_norm10_2 = nn.BatchNorm2d(cfg[v])

        v += 1
        conv11_1 = nn.Conv2d(in_channels=cfg[v - 1], out_channels=cfg[v], kernel_size=1, stride=1, padding=0)
        batch_norm11_1 = nn.BatchNorm2d(cfg[v])

        v += 1
        conv11_2 = nn.Conv2d(in_channels=cfg[v - 1], out_channels=cfg[v], kernel_size=3, stride=1, padding=0)
        batch_norm11_2 = nn.BatchNorm2d(cfg[v])

        relu = nn.ReLU()
        if batch_norm:
            self.block8_1 = nn.Sequential(conv8_1, batch_norm8_1, relu)
            self.block8_2 = nn.Sequential(conv8_2, batch_norm8_2, relu)
            self.block9_1 = nn.Sequential(conv9_1, batch_norm9_1, relu)
            self.block9_2 = nn.Sequential(conv9_2, batch_norm9_2, relu)
            self.block10_1 = nn.Sequential(conv10_1, batch_norm10_1, relu)
            self.block10_2 = nn.Sequential(conv10_2, batch_norm10_2, relu)
            self.block11_1 = nn.Sequential(conv11_1, batch_norm11_1, relu)
            self.block11_2 = nn.Sequential(conv11_2, batch_norm11_2, relu)
        else:
            self.block8_1 = nn.Sequential(conv8_1, relu)
            self.block8_2 = nn.Sequential(conv8_2, relu)
            self.block9_1 = nn.Sequential(conv9_1, relu)
            self.block9_2 = nn.Sequential(conv9_2, relu)
            self.block10_1 = nn.Sequential(conv10_1, relu)
            self.block10_2 = nn.Sequential(conv10_2, relu)
            self.block11_1 = nn.Sequential(conv11_1, relu)
            self.block11_2 = nn.Sequential(conv11_2, relu)

    @autocast()
    def forward(self, x):
        out8_1 = self.block8_1(x)
        out8_2 = self.block8_2(out8_1)
        out9_1 = self.block9_1(out8_2)
        out9_2 = self.block9_2(out9_1)
        out10_1 = self.block10_1(out9_2)
        out10_2 = self.block10_2(out10_1)
        out11_1 = self.block11_1(out10_2)
        out11_2 = self.block11_2(out11_1)
        return [out8_2, out9_2, out10_2, out11_2]


if __name__ == "__main__":
    neck = {
        '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
        '512': [],
    }
    neck = SSDNecks(neck['300'], 1024)
    print(neck)
