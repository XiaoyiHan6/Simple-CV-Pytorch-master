import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.path import CheckPoints
from torch.cuda.amp import autocast

__all__ = ['googlenet']
models_urls = {
    #  'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
    'googlenet': '{}/googlenet-1378be20.pth'.format(CheckPoints),
}


def GoogLeNet(num_classes, pretrained, aux_logits=False, init_weights=False, **kwargs):
    model = googlenet(num_classes=num_classes, aux_logits=aux_logits, init_weights=init_weights, **kwargs)
    if pretrained:
        # if you want to use cpu, you should modify map_loaction=torch.device("cpu")
        pretrained_models = torch.load(models_urls['googlenet'], map_location=torch.device("cuda:0"))
        # transfer learning
        # Inception.branch3.1.conv(kernel_size=3) to Inception.branch3.1.conv(kernel_size=5)
        del pretrained_models['inception3a.branch3.1.conv.weight']
        del pretrained_models['inception3b.branch3.1.conv.weight']
        del pretrained_models['inception4a.branch3.1.conv.weight']
        del pretrained_models['inception4b.branch3.1.conv.weight']
        del pretrained_models['inception4c.branch3.1.conv.weight']
        del pretrained_models['inception4d.branch3.1.conv.weight']
        del pretrained_models['inception4e.branch3.1.conv.weight']
        del pretrained_models['inception5a.branch3.1.conv.weight']
        del pretrained_models['inception5b.branch3.1.conv.weight']
        model.load_state_dict(pretrained_models, strict=False)
    return model


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.relu = nn.ReLU()

    @autocast()
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels=in_channels, out_channels=ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels=in_channels, out_channels=ch3x3red, kernel_size=1),
            BasicConv2d(in_channels=ch3x3red, out_channels=ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels=in_channels, out_channels=ch5x5red, kernel_size=1),
            BasicConv2d(in_channels=ch5x5red, out_channels=ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels=in_channels, out_channels=pool_proj, kernel_size=1)
        )

    @autocast()
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        # aux1 input: 14 * 14 * 512 -> 4 * 4 * 512
        # aux2 input: 14 * 14 * 528 -> 4 * 4 * 528
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        # aux1 4 * 4 * 512 -> 4 * 4 * 128
        # aux2 4 * 4 * 528 -> 4 * 4 * 128
        self.conv = BasicConv2d(in_channels=in_channels, out_channels=128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    @autocast()
    def forward(self, x):
        x = self.averagePool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        return x


class googlenet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=False, init_weights=False):
        super(googlenet, self).__init__()
        self.aux_logits = aux_logits

        # input 224 * 224 * 3 -> 112 * 112 * 64
        self.conv1 = BasicConv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        # 112 * 112 * 64 -> 56 * 56 * 64
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # 56 * 56 * 64 -> 56 * 56 * 64
        self.conv2 = BasicConv2d(in_channels=64, out_channels=64, kernel_size=1)
        # 56 * 56 * 64 -> 56 * 56 * 192
        self.conv3 = BasicConv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1)
        # 56 * 56 * 192 -> 28 * 28 * 192
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # 28 * 28 * 192 -> 28 * 28 * 256
        self.inception3a = Inception(in_channels=192, ch1x1=64, ch3x3red=96, ch3x3=128, ch5x5red=16, ch5x5=32,
                                     pool_proj=32)
        # 28 * 28 * 256 -> 28 * 28 * 480
        self.inception3b = Inception(in_channels=256, ch1x1=128, ch3x3red=128, ch3x3=192, ch5x5red=32, ch5x5=96,
                                     pool_proj=64)
        # 28 * 28 * 480 -> 14 * 14 * 480
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # 14 * 14 * 480 -> 512 * 14 * 14
        self.inception4a = Inception(in_channels=480, ch1x1=192, ch3x3red=96, ch3x3=208, ch5x5red=16, ch5x5=48,
                                     pool_proj=64)
        # 512 * 14 * 14 -> 512 * 14 * 14
        self.inception4b = Inception(in_channels=512, ch1x1=160, ch3x3red=112, ch3x3=224, ch5x5red=24, ch5x5=64,
                                     pool_proj=64)
        # 512 * 14 * 14 -> 512 * 14 * 14
        self.inception4c = Inception(in_channels=512, ch1x1=128, ch3x3red=128, ch3x3=256, ch5x5red=24, ch5x5=64,
                                     pool_proj=64)
        # 512 * 14 * 14 -> 528 * 14 * 14
        self.inception4d = Inception(in_channels=512, ch1x1=112, ch3x3red=144, ch3x3=288, ch5x5red=32, ch5x5=64,
                                     pool_proj=64)
        # 14 * 14 * 528 -> 14 * 14 * 832
        self.inception4e = Inception(in_channels=528, ch1x1=256, ch3x3red=160, ch3x3=320, ch5x5red=32, ch5x5=128,
                                     pool_proj=128)
        # 14 * 14 * 832 -> 7 * 7 * 832
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # 7 * 7 * 832 -> 7 * 7 * 832
        self.inception5a = Inception(in_channels=832, ch1x1=256, ch3x3red=160, ch3x3=320, ch5x5red=32, ch5x5=128,
                                     pool_proj=128)
        # 7 * 7 * 832 -> 7 * 7 * 1024
        self.inception5b = Inception(in_channels=832, ch1x1=384, ch3x3red=192, ch3x3=384, ch5x5red=48, ch5x5=128,
                                     pool_proj=128)
        if self.aux_logits:
            self.aux1 = InceptionAux(in_channels=512, num_classes=num_classes)
            self.aux2 = InceptionAux(in_channels=528, num_classes=num_classes)
        # 1 * 1 * 1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    @autocast()
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)

        if self.training and self.aux_logits:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        if self.training and self.aux_logits:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        if self.training and self.aux_logits:
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
