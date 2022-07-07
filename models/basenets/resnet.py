import torch
import torch.nn as nn
from utils.path import CheckPoints
from torch.cuda.amp import autocast
from torchvision.models.resnet import resnext101_32x8d

__all__ = [
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'resnext50_32x4d',
    'resnext101_32x8d'
]
# if your network is limited, you can download them, and put them into CheckPoints(my Project:Simple-CV-Pytorch-master/checkpoints/).
model_urls = {
    # 'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet18': '{}/resnet18-5c106cde.pth'.format(CheckPoints),
    # 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet34': '{}/resnet34-333f7ec4.pth'.format(CheckPoints),
    # 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet50': '{}/resnet50-19c8e357.pth'.format(CheckPoints),
    # 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet101': '{}/resnet101-5d3b4d8f.pth'.format(CheckPoints),
    # 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet152': '{}/resnet152-b121ed2d.pth'.format(CheckPoints),
    # 'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext50_32x4d': '{}/resnext50_32x4d-7cdf4587.pth'.format(CheckPoints),
    # 'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'resnext101_32x8d': '{}/resnext101_32x8d-8ba56ff5.pth'.format(CheckPoints)
}


def resnet_(arch, block, block_num, num_classes, pretrained, include_top, **kwargs):
    model = resnet(block=block, blocks_num=block_num, num_classes=num_classes, include_top=include_top, **kwargs)
    # if you're training for the first time, no pretrained is required!
    if pretrained:
        # if you want to use cpu, you should modify map_loaction=torch.device("cpu")
        pretrained_models = torch.load(model_urls[arch], map_location=torch.device("cuda:0"))
        # transfer learning
        # if you want to train your own dataset
        # del pretrained_models['module.classifier.bias']
        model.load_state_dict(pretrained_models, strict=False)
    return model


# 18-layer, 34-layer
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    @autocast()
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# 50-layer, 101-layer, 152-layer
class Bottleneck(nn.Module):
    """
    self.conv1(kernel_size=1,stride=2)
    self.conv2(kernel_size=3,stride=1)

    to

    self.conv1(kernel_size=1,stride=1)
    self.conv2(kernel_size=3,stride=2)

    acc: up 0.5%
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channels * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=1,
                               stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=width, kernel_size=3,
                               groups=groups, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel__size=1,
                               stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample

    @autocast()
    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class resnet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000,
                 include_top=True, groups=1, width_per_group=64):
        super(resnet, self).__init__()
        self.include_top = include_top
        self.in_channels = 64
        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channels, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion))
        layers = []
        layers.append(block(in_channels=self.in_channels, out_channels=channels, downsample=downsample,
                            stride=stride, groups=self.groups, width_per_group=self.width_per_group))
        self.in_channels = channels * block.expansion

        for _ in range(1, block_num):
            layers.append(
                block(in_channels=self.in_channels, out_channels=channels,
                      groups=self.groups, width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    @autocast()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = self.flatten(x)
            x = self.fc(x)
        return x


def resnet18(num_classes=1000, pretrained=False, include_top=True):
    return resnet_('resnet18', BasicBlock, [2, 2, 2, 2], num_classes, pretrained, include_top)


def resnet34(num_classes=1000, pretrained=False, include_top=True):
    return resnet_('resnet34', BasicBlock, [3, 4, 6, 3], num_classes, pretrained, include_top)


def resnet50(num_classes=1000, pretrained=False, include_top=True):
    return resnet_('resnet50', Bottleneck, [3, 4, 6, 3], num_classes, pretrained, include_top)


def resnet101(num_classes=1000, pretrained=False, include_top=True):
    return resnet_('resnet101', Bottleneck, [3, 4, 23, 3], num_classes, pretrained, include_top)


def resnet152(num_classes=1000, pretrained=False, include_top=True):
    return resnet_('resnet152', Bottleneck, [3, 8, 36, 3], num_classes, pretrained, include_top)


def resnext50_32x4d(num_classes=1000, pretrained=False, include_top=True):
    groups = 32
    width_per_group = 4
    return resnet_('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], include_top,
                   num_classes=num_classes, pretrained=pretrained,
                   groups=groups, width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, pretrianed=False, include_top=True):
    groups = 32
    width_per_group = 8
    return resnet_('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], include_top,
                   num_classes=num_classes, pretrianed=pretrianed,
                   groups=groups, width_per_group=width_per_group)
