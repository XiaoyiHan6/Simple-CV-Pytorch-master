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
        in_channels = i
        v = 0
        conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=cfg[v], kernel_size=3, stride=1, padding=1)
        batch_norm1_1 = nn.BatchNorm2d(cfg[v])

        v += 1
        conv1_2 = nn.Conv2d(in_channels=cfg[v - 1], out_channels=cfg[v], kernel_size=3, stride=1, padding=1)
        batch_norm1_2 = nn.BatchNorm2d(cfg[v])

        v += 1
        pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        v += 1
        conv2_1 = nn.Conv2d(in_channels=cfg[v - 2], out_channels=cfg[v], kernel_size=3, stride=1, padding=1)
        batch_norm2_1 = nn.BatchNorm2d(cfg[v])

        v += 1
        conv2_2 = nn.Conv2d(in_channels=cfg[v - 1], out_channels=cfg[v], kernel_size=3, stride=1, padding=1)
        batch_norm2_2 = nn.BatchNorm2d(cfg[v])

        v += 1
        pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        v += 1
        conv3_1 = nn.Conv2d(in_channels=cfg[v - 2], out_channels=cfg[v], kernel_size=3, stride=1, padding=1)
        batch_norm3_1 = nn.BatchNorm2d(cfg[v])

        v += 1
        conv3_2 = nn.Conv2d(in_channels=cfg[v - 1], out_channels=cfg[v], kernel_size=3, stride=1, padding=1)
        batch_norm3_2 = nn.BatchNorm2d(cfg[v])

        v += 1
        conv3_3 = nn.Conv2d(in_channels=cfg[v - 1], out_channels=cfg[v], kernel_size=3, stride=1, padding=1)
        batch_norm3_3 = nn.BatchNorm2d(cfg[v])

        v += 1
        pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)

        v += 1
        conv4_1 = nn.Conv2d(in_channels=cfg[v - 2], out_channels=cfg[v], kernel_size=3, stride=1, padding=1)
        batch_norm4_1 = nn.BatchNorm2d(cfg[v])

        v += 1
        conv4_2 = nn.Conv2d(in_channels=cfg[v - 1], out_channels=cfg[v], kernel_size=3, stride=1, padding=1)
        batch_norm4_2 = nn.BatchNorm2d(cfg[v])

        v += 1
        conv4_3 = nn.Conv2d(in_channels=cfg[v - 1], out_channels=cfg[v], kernel_size=3, stride=1, padding=1)
        batch_norm4_3 = nn.BatchNorm2d(cfg[v])

        v += 1
        pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        v += 1
        conv5_1 = nn.Conv2d(in_channels=cfg[v - 2], out_channels=cfg[v], kernel_size=3, stride=1, padding=1)
        batch_norm5_1 = nn.BatchNorm2d(cfg[v])

        v += 1
        conv5_2 = nn.Conv2d(in_channels=cfg[v - 1], out_channels=cfg[v], kernel_size=3, stride=1, padding=1)
        batch_norm5_2 = nn.BatchNorm2d(cfg[v])
        v += 1
        conv5_3 = nn.Conv2d(in_channels=cfg[v - 1], out_channels=cfg[v], kernel_size=3, stride=1, padding=1)
        batch_norm5_3 = nn.BatchNorm2d(cfg[v])

        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
        conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=6, dilation=6)
        batch_norm6 = nn.BatchNorm2d(1024)
        conv7 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1)
        batch_norm7 = nn.BatchNorm2d(1024)
        relu = nn.ReLU()
        if batch_norm:
            self.block1_1 = nn.Sequential(conv1_1, batch_norm1_1, relu)
            self.block1_2 = nn.Sequential(conv1_2, batch_norm1_2, relu)
            self.pool1 = nn.Sequential(pool1)

            self.block2_1 = nn.Sequential(conv2_1, batch_norm2_1, relu)
            self.block2_2 = nn.Sequential(conv2_2, batch_norm2_2, relu)
            self.pool2 = nn.Sequential(pool2)

            self.block3_1 = nn.Sequential(conv3_1, batch_norm3_1, relu)
            self.block3_2 = nn.Sequential(conv3_2, batch_norm3_2, relu)
            self.block3_3 = nn.Sequential(conv3_3, batch_norm3_3, relu)
            self.pool3 = nn.Sequential(pool3)

            self.block4_1 = nn.Sequential(conv4_1, batch_norm4_1, relu)
            self.block4_2 = nn.Sequential(conv4_2, batch_norm4_2, relu)
            self.block4_3 = nn.Sequential(conv4_3, batch_norm4_3, relu)
            self.pool4 = nn.Sequential(pool4)

            self.block5_1 = nn.Sequential(conv5_1, batch_norm5_1, relu)
            self.block5_2 = nn.Sequential(conv5_2, batch_norm5_2, relu)
            self.block5_3 = nn.Sequential(conv5_3, batch_norm5_3, relu)
            self.pool5 = nn.Sequential(pool5)

            self.block6 = nn.Sequential(conv6, batch_norm6, relu)
            self.block7 = nn.Sequential(conv7, batch_norm7, relu)
        else:
            self.block1_1 = nn.Sequential(conv1_1, relu)
            self.block1_2 = nn.Sequential(conv1_2, relu)
            self.pool1 = nn.Sequential(pool1)

            self.block2_1 = nn.Sequential(conv2_1, relu)
            self.block2_2 = nn.Sequential(conv2_2, relu)
            self.pool2 = nn.Sequential(pool2)

            self.block3_1 = nn.Sequential(conv3_1, relu)
            self.block3_2 = nn.Sequential(conv3_2, relu)
            self.block3_3 = nn.Sequential(conv3_3, relu)
            self.pool3 = nn.Sequential(pool3)

            self.block4_1 = nn.Sequential(conv4_1, relu)
            self.block4_2 = nn.Sequential(conv4_2, relu)
            self.block4_3 = nn.Sequential(conv4_3, relu)
            self.pool4 = nn.Sequential(pool4)

            self.block5_1 = nn.Sequential(conv5_1, relu)
            self.block5_2 = nn.Sequential(conv5_2, relu)
            self.block5_3 = nn.Sequential(conv5_3, relu)
            self.pool5 = nn.Sequential(pool5)

            self.block6 = nn.Sequential(conv6, relu)
            self.block7 = nn.Sequential(conv7, relu)

    @autocast()
    def forward(self, x):
        x = self.block1_1(x)
        x = self.block1_2(x)
        x = self.pool1(x)

        x = self.block2_2(x)
        x = self.block2_2(x)
        x = self.pool2(x)

        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.pool3(x)

        x = self.block4_1(x)
        x = self.block4_2(x)
        out1 = self.block4_3(x)
        x = self.pool4(out1)

        x = self.block5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.pool5(x)

        x = self.block6(x)
        out2 = self.block7(x)

        return [out1, out2]


if __name__ == "__main__":
    backbone = {
        '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
        '512': [],
    }
    layers_1 = VggNetBackbone(backbone['300'], 3, batch_norm=True)
    print(layers_1)
    layers_2 = VggNetBackbone(backbone['300'], 3)
    print(layers_2)
