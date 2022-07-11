import torch
import torch.nn as nn
from functools import partial
from typing import List
from utils.path import CheckPoints
from torch.cuda.amp import autocast
from torch.nn import functional as F

__all__ = ['mobilenet_v3_large', 'mobilenet_v3_small']

models_urls = {
    #  'mobilenet_v3_small' : 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth'
    'mobilenet_v3_small': '{}/mobilenet_v3_small-047dcff4.pth'.format(CheckPoints),
    # 'mobilenet_v3_large' : 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth'
    'mobilenet_v3_large': '{}/mobilenet_v3_large-8738ca79.pth'.format(CheckPoints)
}


def MobileNet_v3(pretrained=False, num_classes=1000, init_weights=False, type='small'):
    if type == 'small':
        return mobilenet_v3_small(pretrained=pretrained, num_classes=num_classes, init_weights=init_weights)
    elif type == 'large':
        return mobilenet_v3_large(pretrained=pretrained, num_classes=num_classes, init_weights=init_weights)
    else:
        raise ValueError("Unsupported MobileNet V3 Type!")


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNActivation(nn.Sequential):
    def __init__(self, norm_layer, activation_layer, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer())


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c, squeeze_factor=4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(in_channels=input_c, out_channels=squeeze_c, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels=squeeze_c, out_channels=input_c, kernel_size=1)

    @autocast()
    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale)
        return scale * x


class InvertedResidualConfig:
    def __init__(self, input_c, kernel, expanded_c, out_c, use_se, activation, stride, width_multi):
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        # whether using h-swish activation
        self.use_hs = activation == 'HS'
        self.stride = stride

    @staticmethod
    def adjust_channels(channels, width_multi):
        return _make_divisible(channels * width_multi, 8)


class InvertedResidual(nn.Module):
    def __init__(self, cnf, norm_layer):
        super(InvertedResidual, self).__init__()
        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers = []

        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_c != cnf.input_c:
            layers.append(
                ConvBNActivation(norm_layer=norm_layer, activation_layer=activation_layer,
                                 in_planes=cnf.input_c, out_planes=cnf.expanded_c, kernel_size=1))
            # depthwise
            layers.append(
                ConvBNActivation(norm_layer=norm_layer, activation_layer=activation_layer,
                                 in_planes=cnf.expanded_c, out_planes=cnf.expanded_c,
                                 kernel_size=cnf.kernel, stride=cnf.stride,
                                 groups=cnf.expanded_c))

        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_c))

        # project
        layers.append(
            ConvBNActivation(norm_layer=norm_layer, activation_layer=nn.Identity,
                             in_planes=cnf.expanded_c, out_planes=cnf.out_c,
                             kernel_size=1))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


class mobilenet_v3(nn.Module):
    def __init__(self,
                 inverted_residual_setting,
                 last_channels, block, norm_layer, num_classes=1000, init_weights=False):
        super(mobilenet_v3, self).__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty.")

        elif not (isinstance(inverted_residual_setting, List) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers = []

        # building first layer
        firstconv_output_c = inverted_residual_setting[0].input_c
        layers.append(
            ConvBNActivation(norm_layer=norm_layer, activation_layer=nn.Hardswish,
                             in_planes=3, out_planes=firstconv_output_c,
                             kernel_size=3, stride=2))

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_c = inverted_residual_setting[-1].out_c
        lastconv_output_c = 6 * lastconv_input_c
        layers.append(
            ConvBNActivation(norm_layer=norm_layer, activation_layer=nn.Hardswish,
                             in_planes=lastconv_input_c, out_planes=lastconv_output_c,
                             kernel_size=1))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_c, last_channels),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(last_channels, num_classes))
        self.init_weights = init_weights
        if self.init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_v3_large(reduced_tail=False, num_classes=1000, pretrained=False, init_weights=False):
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, 'RE', 1),
        bneck_conf(16, 3, 64, 24, False, 'RE', 2),
        bneck_conf(24, 3, 72, 24, False, 'RE', 1),
        bneck_conf(24, 5, 72, 40, True, 'RE', 2),
        bneck_conf(40, 5, 120, 40, True, 'RE', 1),
        bneck_conf(40, 5, 120, 40, True, 'RE', 1),
        bneck_conf(40, 3, 240, 80, False, 'HS', 2),
        bneck_conf(80, 3, 200, 80, False, 'HS', 1),
        bneck_conf(80, 3, 184, 80, False, 'HS', 1),
        bneck_conf(80, 3, 184, 80, False, 'HS', 1),
        bneck_conf(80, 3, 480, 112, True, 'HS', 1),
        bneck_conf(112, 3, 672, 112, True, 'HS', 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, 'HS', 2),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, 'HS', 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, 'HS', 1),
    ]
    last_channels = adjust_channels(1280 // reduce_divider)
    model = mobilenet_v3(inverted_residual_setting=inverted_residual_setting, last_channels=last_channels,
                         block=None, norm_layer=None, num_classes=num_classes, init_weights=init_weights)
    if pretrained:
        # if you want to use cpu, you should modify map_loaction=torch.device("cpu")
        pretrained_models = torch.load(models_urls['mobilenet_v3_large'], map_location=torch.device("cuda:0"))
        # transfer learning
        del pretrained_models['features.1.block.0.0.weight']
        model.load_state_dict(pretrained_models, strict=False)
    return model


def mobilenet_v3_small(reduced_tail=False, num_classes=1000, pretrained=False, init_weights=False):
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, True, 'RE', 2),
        bneck_conf(16, 3, 72, 24, False, 'RE', 2),
        bneck_conf(24, 3, 88, 24, False, 'RE', 1),
        bneck_conf(24, 5, 96, 40, True, 'HS', 2),
        bneck_conf(40, 5, 240, 40, True, 'HS', 1),
        bneck_conf(40, 5, 240, 40, True, 'HS', 1),
        bneck_conf(40, 5, 120, 48, True, 'HS', 1),
        bneck_conf(48, 5, 144, 48, True, 'HS', 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, 'HS', 2),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, 'HS', 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, 'HS', 1),
    ]
    last_channels = adjust_channels(1024 // reduce_divider)

    model = mobilenet_v3(inverted_residual_setting=inverted_residual_setting, last_channels=last_channels,
                         block=None, norm_layer=None, num_classes=num_classes, init_weights=init_weights)
    if pretrained:
        # if you want to use cpu, you should modify map_loaction=torch.device("cpu")
        pretrained_models = torch.load(models_urls['mobilenet_v3_small'], map_location=torch.device("cuda:0"))
        model.load_state_dict(pretrained_models, strict=False)
    return model
