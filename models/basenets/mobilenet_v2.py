import torch
import torch.nn as nn
from utils.path import CheckPoints
from torch.cuda.amp import autocast

__all__ = ['mobilenet_v2']
models_urls = {
    # "mobilenet_v2": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
    'mobilenet_v2': '{}/mobilenet_v2-b0353104.pth'.format(CheckPoints),
}


def MobileNet_v2(num_classes=1000, pretrained=False, init_weights=False, **kwargs):
    model = mobilenet_v2(num_classes=num_classes, init_weights=init_weights, **kwargs)
    if pretrained:
        # if you want to use cpu, you should modify map_loaction=torch.device("cpu")
        pretrained_models = torch.load(models_urls['mobilenet_v2'], map_location=torch.device("cuda:0"))
        model.load_state_dict(pretrained_models, strict=False)
    return model


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6())


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channels = in_channels * expand_ratio
        self.use_shortcut = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(in_channels=hidden_channels, out_channels=hidden_channels, stride=stride,
                       groups=hidden_channels),
            # 1x1 pointwise conv(liner)
            nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers)

    @autocast()
    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)

        else:
            return self.conv(x)


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class mobilenet_v2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8, init_weights=False):
        super(mobilenet_v2, self).__init__()
        block = InvertedResidual
        input_channels = _make_divisible(32 * alpha, round_nearest)
        last_channels = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t,c,n,s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(in_channels=3, out_channels=input_channels, stride=2))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channels = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(in_channels=input_channels, out_channels=output_channels, stride=stride, expand_ratio=t))
                input_channels = output_channels
        # building last several layers
        features.append(ConvBNReLU(in_channels=input_channels, out_channels=last_channels, kernel_size=1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channels, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    @autocast()
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x