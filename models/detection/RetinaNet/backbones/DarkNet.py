import torch
import torch.nn as nn

__all__ = [
    'darknettiny',
    'darknet19',
    'darknet53',
]


class DarkNet(nn.Module):
    def __init__(self, darknet_type='darknet19'):
        super(DarkNet, self).__init__()
        self.darknet_type = darknet_type
        if darknet_type == 'darknettiny':
            self.model = darknettiny()
        elif darknet_type == 'darknet19':
            self.model = darknet19()
        elif darknet_type == 'darknet53':
            self.model = darknet53()

    def forward(self, x):
        out = self.model(x)
        return out


class ActBlock(nn.Module):
    def __init__(self, act_type='leakyrelu', inplace=True):
        super(ActBlock, self).__init__()
        assert act_type in ['silu', 'relu', 'leakyrelu'], \
            "Unsupported activation function!"
        if act_type == 'silu':
            self.act = nn.SiLU(inplace=inplace)
        elif act_type == 'relu':
            self.act = nn.ReLU(inplace=inplace)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(0.1, inplace=inplace)

    def forward(self, x):
        x = self.act(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, padding, groups=1, has_bn=True, has_act=True,
                 act_type='leakyrelu'):
        super(ConvBlock, self).__init__()
        bias = False if has_bn else True

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=bias),
            nn.BatchNorm2d(planes) if has_bn else nn.Sequential(),
            ActBlock(act_type=act_type, inplace=True) if has_act else nn.Sequential()

        )

    def forward(self, x):
        x = self.layer(x)
        return x


class DarkNetTiny(nn.Module):
    def __init__(self, act_type='leakyrelu'):
        super(DarkNetTiny, self).__init__()
        self.conv1 = ConvBlock(inplanes=3, planes=16, kernel_size=3, stride=1, padding=1, groups=1, has_bn=True,
                               has_act=True, act_type=act_type)
        self.conv2 = ConvBlock(inplanes=16, planes=32, kernel_size=3, stride=1, padding=1, groups=1, has_bn=True,
                               has_act=True, act_type=act_type)
        self.conv3 = ConvBlock(inplanes=32, planes=64, kernel_size=3, stride=1, padding=1, groups=1, has_bn=True,
                               has_act=True, act_type=act_type)
        self.conv4 = ConvBlock(inplanes=64, planes=128, kernel_size=3, stride=1, padding=1, groups=1, has_bn=True,
                               has_act=True, act_type=act_type)
        self.conv5 = ConvBlock(inplanes=128, planes=256, kernel_size=3, stride=1, padding=1, groups=1, has_bn=True,
                               has_act=True, act_type=act_type)
        self.conv6 = ConvBlock(inplanes=256, planes=512, kernel_size=3, stride=1, padding=1, groups=1, has_bn=True,
                               has_act=True, act_type=act_type)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.zeropad = nn.ZeroPad2d((0, 1, 0, 1))
        self.last_maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.out_channels = [64, 128, 256]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.maxpool(x)

        C3 = self.conv3(x)
        C3 = self.maxpool(C3)

        C4 = self.conv4(C3)
        C4 = self.maxpool(C4)  # 128

        C5 = self.conv5(C4)
        C5 = self.maxpool(C5)  # 256

        del x
        return [C3, C4, C5]


class D19Block(nn.Module):
    def __init__(self, inplanes, planes, layer_num, use_maxpool=False, act_type='leakyrelu'):
        super(D19Block, self).__init__()
        self.use_maxpool = use_maxpool
        layers = []
        for i in range(0, layer_num):
            if i % 2 == 0:
                layers.append(
                    ConvBlock(inplanes=inplanes, planes=planes, kernel_size=3, stride=1, padding=1, groups=1,
                              has_bn=True, has_act=True, act_type=act_type))
            else:
                layers.append(
                    ConvBlock(inplanes=planes, planes=inplanes, kernel_size=1, stride=1, padding=0, groups=1,
                              has_bn=True, has_act=True, act_type=act_type))
        self.D19Block = nn.Sequential(*layers)
        if self.use_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.D19Block(x)

        if self.use_maxpool:
            x = self.maxpool(x)
        return x


class DarkNet19(nn.Module):
    def __init__(self, act_type='leakyrelu'):
        super(DarkNet19, self).__init__()

        self.layer1 = ConvBlock(inplanes=3, planes=32, kernel_size=3, stride=1, padding=1, groups=1, has_bn=True,
                                has_act=True, act_type=act_type)
        self.layer2 = D19Block(inplanes=32, planes=64, layer_num=1, use_maxpool=True, act_type=act_type)
        self.layer3 = D19Block(inplanes=64, planes=128, layer_num=3, use_maxpool=True, act_type=act_type)
        self.layer4 = D19Block(inplanes=128, planes=256, layer_num=3, use_maxpool=True, act_type=act_type)
        self.layer5 = D19Block(inplanes=256, planes=512, layer_num=5, use_maxpool=True, act_type=act_type)
        self.layer6 = D19Block(inplanes=512, planes=1024, layer_num=5, use_maxpool=False, act_type=act_type)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.out_channels = [128, 256, 512]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)

        C3 = self.layer3(x)
        C4 = self.layer4(C3)
        C5 = self.layer5(C4)

        del x
        return [C3, C4, C5]


# conv*2+residual
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvBlock(inplanes=inplanes, planes=planes, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBlock(inplanes=planes, planes=planes * 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        out += x
        del x
        return out


class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53, self).__init__()
        self.conv1 = ConvBlock(inplanes=3, planes=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(inplanes=32, planes=64, kernel_size=3, stride=2, padding=1)

        self.block1 = nn.Sequential(
            BasicBlock(inplanes=64, planes=32),
            ConvBlock(inplanes=64, planes=128, kernel_size=3, stride=2, padding=1)
        )  # 128

        self.block2 = nn.Sequential(
            BasicBlock(inplanes=128, planes=64),
            BasicBlock(inplanes=128, planes=64),
            ConvBlock(inplanes=128, planes=256, kernel_size=3, stride=2, padding=1)
        )  # 256

        self.block3 = nn.Sequential(
            BasicBlock(inplanes=256, planes=128),
            BasicBlock(inplanes=256, planes=128),
            BasicBlock(inplanes=256, planes=128),
            BasicBlock(inplanes=256, planes=128),
            BasicBlock(inplanes=256, planes=128),
            BasicBlock(inplanes=256, planes=128),
            BasicBlock(inplanes=256, planes=128),
            BasicBlock(inplanes=256, planes=128),
            ConvBlock(inplanes=256, planes=512, kernel_size=3, stride=2, padding=1)
        )  # 512

        self.block4 = nn.Sequential(
            BasicBlock(inplanes=512, planes=256),
            BasicBlock(inplanes=512, planes=256),
            BasicBlock(inplanes=512, planes=256),
            BasicBlock(inplanes=512, planes=256),
            BasicBlock(inplanes=512, planes=256),
            BasicBlock(inplanes=512, planes=256),
            BasicBlock(inplanes=512, planes=256),
            BasicBlock(inplanes=512, planes=256),
            ConvBlock(inplanes=512, planes=1024, kernel_size=3, stride=2, padding=1)
        )  # 1024

        self.block5 = nn.Sequential(
            BasicBlock(inplanes=1024, planes=512),
            BasicBlock(inplanes=1024, planes=512),
            BasicBlock(inplanes=1024, planes=512),
            BasicBlock(inplanes=1024, planes=512)
        )

        self.out_channels = [256, 512, 1024]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.block1(x)
        C3 = self.block2(x)
        C4 = self.block3(C3)
        C5 = self.block4(C4)
        del x
        return [C3, C4, C5]


def darknettiny(**kwargs):
    model = DarkNetTiny(**kwargs)
    return model


def darknet19(**kwargs):
    model = DarkNet19(**kwargs)
    return model


def darknet53(**kwargs):
    model = DarkNet53(**kwargs)
    return model


if __name__ == '__main__':
    x = torch.randn([8, 3, 512, 512])
    darknet = DarkNet(darknet_type='darknet53')
    [C3, C4, C5] = darknet(x)
    print("C3.shape:{}".format(C3.shape))
    print("C4.shape:{}".format(C4.shape))
    print("C5.shape:{}".format(C5.shape))

    # DarkNet53
    # C3.shape: torch.Size([8, 256, 64, 64])
    # C4.shape: torch.Size([8, 512, 32, 32])
    # C5.shape: torch.Size([8, 1024, 16, 16])

    # DarkNet19
    # C3.shape: torch.Size([8, 128, 64, 64])
    # C4.shape: torch.Size([8, 256, 32, 32])
    # C5.shape: torch.Size([8, 512, 16, 16])

    # DarkNetTiny
    # C3.shape: torch.Size([8, 64, 64, 64])
    # C4.shape: torch.Size([8, 128, 32, 32])
    # C5.shape: torch.Size([8, 256, 16, 16])
