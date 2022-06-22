import torch
from torch import nn
from utils.path import CheckPoints
__all__ = [
    'vgg16',
]
model_urls = {
    # https://download.pytorch.org/models/vgg16-397923af.pth'
    'vgg16': '{}/vgg16-397923af.pth'.format(CheckPoints)
}


def _vgg(arch, num_classes, pretrained, progress, **kwargs):
    model = vgg(num_classes=num_classes, **kwargs)
    # if you're training for the first time, no pretrained is required!
    pretrained_models = torch.load(model_urls["vgg" + arch])
    # transfer learning
    # if you want to train your own dataset
    del pretrained_models['module.classifier.7.bias']

    if pretrained:
        model.load_state_dict(pretrained_models, strict=False)
    return model


def vgg16(num_classes, pretrained=False, progress=True, **kwargs):
    return _vgg('16', num_classes, pretrained, progress, **kwargs)


class vgg(nn.Module):
    def __init__(self, num_classes=2):
        super(vgg, self).__init__()
        self.layer1 = nn.Sequential(
            # 1
            # 64*64*3 -> 64*64*64
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 2
            # 64*64*64 -> 32*32*64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            # 3
            # 32*32*64 -> 32*32*128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 4
            # 32*32*128 -> 16*16*128
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            # 5
            # 16*16*128 -> 16*16*256
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 6
            # 16*16*256 -> 16*16*256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 7
            # 16*16*256 -> 8*8*256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            # 8
            # 8*8*256 -> 8*8*512
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 9
            # 8*8*512 -> 8*8*512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 10
            # 8*8*512 -> 4*4*512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer5 = nn.Sequential(
            # 11
            # 4*4*512 -> 4*4*512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 12
            # 4*4*512 -> 4*4*512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 13
            # 4*4*512 -> 2*2*512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
        )

        self.fc = nn.Sequential(
            # 14
            nn.Flatten(),
            nn.Linear(2 * 2 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # 15
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        # 16
        self.classifier = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.classifier(x)
        return x
