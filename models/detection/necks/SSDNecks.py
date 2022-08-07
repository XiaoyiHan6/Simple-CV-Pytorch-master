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
        layers = []
        # 1024
        in_channels = i
        # flag choose kernel_size = 1 or 3
        flag = False
        for k, v in enumerate(cfg):
            if in_channels != 'S':
                if v == 'S':
                    conv2d_1 = nn.Conv2d(in_channels=in_channels, out_channels=cfg[k + 1],
                                         kernel_size=(1, 3)[flag], stride=2, padding=1)
                    if batch_norm:
                        layers += [conv2d_1, nn.BatchNorm2d(cfg[k + 1]), nn.ReLU()]
                    else:
                        layers += [conv2d_1, nn.ReLU()]
                else:
                    conv2d_2 = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=(1, 3)[flag])
                    if batch_norm:
                        layers += [conv2d_2, nn.BatchNorm2d(v), nn.ReLU()]
                    else:
                        layers += [conv2d_2, nn.ReLU()]

                flag = not flag
            in_channels = v
        self.neck = nn.Sequential(*layers)

    @autocast()
    def forward(self, x):
        x = self.neck(x)
        return x


class SSDNecks_modify(nn.Module):
    def __init__(self, cfg, i, batch_norm=False):
        super(SSDNecks_modify, self).__init__()
        """
        cfg: channels of layer
        i: nput_channels
        batch_norm: whether to use BN
        """
        layers = []
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
            layers += [neck1_1, batch_norm1_1, relu]
            layers += [neck1_2, batch_norm1_2, relu]
            layers += [neck2_1, batch_norm2_1, relu]
            layers += [neck2_2, batch_norm2_2, relu]
            layers += [neck3_1, batch_norm3_1, relu]
            layers += [neck3_2, batch_norm3_2, relu]
            layers += [neck4_1, batch_norm4_1, relu]
            layers += [neck4_2, batch_norm4_2, relu]
        else:
            layers += [neck1_1, relu]
            layers += [neck1_2, relu]
            layers += [neck2_1, relu]
            layers += [neck2_2, relu]
            layers += [neck3_1, relu]
            layers += [neck3_2, relu]
            layers += [neck4_1, relu]
            layers += [neck4_2, relu]
        self.neck = nn.Sequential(*layers)


if __name__ == "__main__":
    neck = {
        '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
        '512': [],
    }
    neck = SSDNecks_modify(neck['300'], 1024)
    print(neck)
