import torch
from torch import nn


class lenet5(nn.Module):
    # cifar: 10, ImageNet: 1000
    def __init__(self, num_classes=1000):
        super(lenet5, self).__init__()
        self.num_classes = num_classes
        self.model = nn.Sequential(
            # input:32 * 32 * 3 -> 28 * 28 * 6
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=0, stride=1),
            nn.ReLU(inplace=True),
            # 28 * 28 * 6 -> 14 * 14 * 6
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # 14 * 14 * 6 -> 10 * 10 * 16
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0, stride=1),
            nn.ReLU(inplace=True),
            # 10 * 10 * 16 -> 5 * 5 * 16
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84))
        self.classifier = nn.Linear(84, self.num_classes),

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x
