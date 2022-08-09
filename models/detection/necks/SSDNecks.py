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
        neck1_1 = nn.Conv2d(in_channels=in_channels, out_channels=cfg[v], kernel_size=1, stride=1)
        batch_norm1_1 = nn.BatchNorm2d(256)

        v += 2
        neck1_2 = nn.Conv2d(in_channels=256, out_channels=cfg[v], kernel_size=3, stride=2, padding=1)
        batch_norm1_2 = nn.BatchNorm2d(512)

        v += 1
        neck2_1 = nn.Conv2d(in_channels=512, out_channels=cfg[v], kernel_size=1, stride=1, padding=0)
        batch_norm2_1 = nn.BatchNorm2d(128)

        v += 2
        neck2_2 = nn.Conv2d(in_channels=128, out_channels=cfg[v], kernel_size=3, stride=2, padding=1)
        batch_norm2_2 = nn.BatchNorm2d(256)

        v += 1
        neck3_1 = nn.Conv2d(in_channels=256, out_channels=cfg[v], kernel_size=1, stride=1, padding=0)
        batch_norm3_1 = nn.BatchNorm2d(128)

        v += 1
        neck3_2 = nn.Conv2d(in_channels=128, out_channels=cfg[v], kernel_size=3, stride=1, padding=0)
        batch_norm3_2 = nn.BatchNorm2d(256)

        v += 1
        neck4_1 = nn.Conv2d(in_channels=256, out_channels=cfg[v], kernel_size=1, stride=1, padding=0)
        batch_norm4_1 = nn.BatchNorm2d(128)

        v += 1
        neck4_2 = nn.Conv2d(in_channels=128, out_channels=cfg[v], kernel_size=3, stride=1, padding=0)
        batch_norm4_2 = nn.BatchNorm2d(256)

        relu = nn.ReLU()
        if batch_norm:
            self.neck1_1 = nn.Sequential(neck1_1, batch_norm1_1, relu)
            self.neck1_2 = nn.Sequential(neck1_2, batch_norm1_2, relu)
            self.neck2_1 = nn.Sequential(neck2_1, batch_norm2_1, relu)
            self.neck2_2 = nn.Sequential(neck2_2, batch_norm2_2, relu)
            self.neck3_1 = nn.Sequential(neck3_1, batch_norm3_1, relu)
            self.neck3_2 = nn.Sequential(neck3_2, batch_norm3_2, relu)
            self.neck4_1 = nn.Sequential(neck4_1, batch_norm4_1, relu)
            self.neck4_2 = nn.Sequential(neck4_2, batch_norm4_2, relu)
        else:
            self.neck1_1 = nn.Sequential(neck1_1, relu)
            self.neck1_2 = nn.Sequential(neck1_2, relu)
            self.neck2_1 = nn.Sequential(neck2_1, relu)
            self.neck2_2 = nn.Sequential(neck2_2, relu)
            self.neck3_1 = nn.Sequential(neck3_1, relu)
            self.neck3_2 = nn.Sequential(neck3_2, relu)
            self.neck4_1 = nn.Sequential(neck4_1, relu)
            self.neck4_2 = nn.Sequential(neck4_2, relu)

    @autocast()
    def forward(self, x):
        out1_1 = self.neck1_1(x)
        out1_2 = self.neck1_2(out1_1)
        out2_1 = self.neck2_1(out1_2)
        out2_2 = self.neck2_2(out2_1)
        out3_1 = self.neck3_1(out2_2)
        out3_2 = self.neck3_2(out3_1)
        out4_1 = self.neck4_1(out3_2)
        out4_2 = self.neck4_2(out4_1)
        return [out1_2, out2_2, out3_2, out4_2]


if __name__ == "__main__":
    neck = {
        '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
        '512': [],
    }
    neck = SSDNecks(neck['300'], 1024)
    print(neck)
