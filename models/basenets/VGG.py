import torch
from torch import nn
from utils.path import CheckPoints

__all__ = [
    'vgg11',
    'vgg13',
    'vgg16',
    'vgg19',
]
# if your network is limited, you can download them, and put them into CheckPoints(my Project:Simple-CV-Pytorch-master/).
model_urls = {
    # 'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg11': '{}/vgg11-bbd30ac9.pth'.format(CheckPoints),
    # 'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg13': '{}/vgg13-c768596a.pth'.format(CheckPoints),
    # 'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16': '{}/vgg16-397923af.pth'.format(CheckPoints),
    # 'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg19': '{}/vgg19-dcbb9e9d.pth'.format(CheckPoints)

}


def vgg_(arch, num_classes, pretrained, init_weights=False, **kwargs):
    cfg = cfgs["vgg" + arch]
    features = make_features(cfg)
    model = VGG(num_classes=num_classes, features=features, init_weights=init_weights, **kwargs)
    # if you're training for the first time, no pretrained is required!
    pretrained_models = torch.load(model_urls["vgg" + arch])
    # transfer learning
    # if you want to train your own dataset
    # del pretrained_models['module.classifier.bias']

    if pretrained:
        model.load_state_dict(pretrained_models, strict=False)
    return model


def vgg11(num_classes, pretrained=False, init_weights=False, **kwargs):
    return vgg_('11', num_classes, pretrained, init_weights, **kwargs)


def vgg13(num_classes, pretrained=False, init_weights=False, **kwargs):
    return vgg_('13', num_classes, pretrained, init_weights, **kwargs)


def vgg16(num_classes, pretrained=False, init_weights=False, **kwargs):
    return vgg_('16', num_classes, pretrained, init_weights, **kwargs)


def vgg19(num_classes, pretrained=False, init_weights=False, **kwargs):
    return vgg_('19', num_classes, pretrained, init_weights, **kwargs)


class VGG(nn.Module):
    # cifar: 10, ImageNet: 1000
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.num_classes = num_classes
        self.features = features
        self.fc = nn.Sequential(

            nn.Flatten(),
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5, inplace=True),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5, inplace=True),
        )

        self.classifier = nn.Linear(4096, self.num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


def make_features(cfgs: list):
    layers = []
    in_channels = 3
    for i in cfgs:
        if i == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, i, kernel_size=3, stride=1, padding=1)
            layers += [conv2d, nn.BatchNorm2d(i), nn.ReLU(inplace=True)]
            in_channels = i
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],

}
print(vgg16(num_classes=1000))
