import torch
from torch import nn
from utils.path import CheckPoints
from torch.cuda.amp import autocast

__all__ = [
    'shufflenet_v2_x0_5',
    'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5',
    'shufflenet_v2_x2_0',
]
model_urls = {
    # 'shufflenet_v2_x0_5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenet_v2_x0_5': '{}/shufflenetv2_x0.5-f707e7126e.pth'.format(CheckPoints),
    # 'shufflenet_v2_x1_0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenet_v2_x1_0': '{}/shufflenetv2_x1-5666bf0f80.pth'.format(CheckPoints),
    # 'shufflenet_v2_x1_5': 'https://download.pytorch.org/models/shufflenetv2_x1_5-3c479a10.pth',
    'shufflenet_v2_x1_5': '{}/shufflenetv2_x1_5-3c479a10.pth'.format(CheckPoints),
    # 'shufflenet_v2_x2_0': 'https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth',
    'shufflenet_v2_x2_0': '{}/shufflenetv2_x2_0-8be3c8ee.pth'.format(CheckPoints),

}


def shufflenet_v2_x(arch, stages_repeats, stages_out_channels, num_classes, pretrained, **kwargs):
    model = ShuffleNetV2(stages_repeats, stages_out_channels, num_classes=num_classes, **kwargs)
    if pretrained:
        pretrained_models = torch.load(model_urls['shufflenet_v2_x' + arch], map_location=torch.device("cuda:0"))
        model.load_state_dict(pretrained_models, strict=False)
    return model


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, group, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)
    return x


class InvertedResidual(nn.Module):
    def __init__(self, input_c, output_c, stride):
        super(InvertedResidual, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.stride = stride
        assert output_c % 2 == 0
        branch_features = output_c // 2
        # << -> x2
        assert (self.stride != 1) or (input_c == branch_features << 1)
        if self.stride == 2:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU()
            )
        else:
            self.branch1 = nn.Sequential()
        self.branch2 = nn.Sequential(
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU()
        )

    @staticmethod
    def depthwise_conv(input_c, ouptput_c, kernel_size, stride, padding=0, bias=False):
        return nn.Conv2d(in_channels=input_c, out_channels=ouptput_c, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias=bias, groups=input_c)

    @autocast()
    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, inverted_residual=InvertedResidual, num_classes=1000):
        super(ShuffleNetV2, self).__init__()
        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        # input RGB image
        input_channels = 3
        output_channels = self._stage_out_channels[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #  Static annotations
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
        self.fc = nn.Linear(output_channels, num_classes)

    @autocast()
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])
        x = self.fc(x)
        return x


def shufflenet_v2_x0_5(num_classes=1000, pretrained=False):
    return shufflenet_v2_x('0_5', stages_repeats=[4, 8, 4], stages_out_channels=[24, 48, 96, 192, 1024],
                           num_classes=num_classes, pretrained=pretrained)


def shufflenet_v2_x1_0(num_classes=1000, pretrianed=False):
    return shufflenet_v2_x('1_0', stages_repeats=[4, 8, 4], stages_out_channels=[24, 116, 232, 464, 1024],
                           num_classes=num_classes, pretrianed=pretrianed)


def shufflenet_v2_x1_5(num_classes=1000, pretrained=False):
    return shufflenet_v2_x('1_5', stages_repeats=[4, 8, 4], stages_out_channels=[24, 176, 352, 704, 1024],
                           num_classes=num_classes, pretrained=pretrained)


def shufflenet_v2_x2_0(num_classes=1000, pretrained=False):
    return shufflenet_v2_x('2_0', stages_repeats=[4, 8, 4], stages_out_channels=[24, 244, 488, 976, 2048],
                           num_classes=num_classes, pretrained=pretrained)
