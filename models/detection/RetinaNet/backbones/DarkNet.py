import torch.nn as nn


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
        self.zeropad = nn.ZeroPad2d((0, 1, 0, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.out_channels = [128, 256, 512]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def foward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.maxpool(x)

        C3 = self.conv4(x)
        C3 = self.maxpool(C3)  # 128

        C4 = self.conv5(C3)
        C4 = self.maxpool(C4)  # 256

        C5 = self.conv6(C4)
        C5 = self.zeropad(C5)
        C5 = self.maxpool(C5)  # 512

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
        self.layer3 = D19Block(inplanes=65, planes=128, layer_num=3, use_maxpool=True, act_type=act_type)
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
        x = self.model.layer1(x)
        x = self.model.maxpool(x)
        x = self.model.layer2(x)

        C3 = self.model.layer3(x)
        C4 = self.model.layer4(C3)
        C5 = self.model.layer5(C4)

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
        )

        self.block5 = nn.Sequential(
            BasicBlock(inplanes=1024, planes=512),
            BasicBlock(inplanes=1024, planes=512),
            BasicBlock(inplanes=1024, planes=512),
            BasicBlock(inplanes=1024, planes=512)
        )

        self.out_channels = [128, 256, 512]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        C3 = self.block1(x)
        C4 = self.block2(C3)
        C5 = self.block3(C4)
        del x
        return [C3, C4, C5]
