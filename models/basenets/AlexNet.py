import torch.nn as nn
from torch.cuda.amp import autocast


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.layer = nn.Sequential(
            # input: 224 * 224 * 3 -> 55 * 55 * (48*2)
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            # 55 * 55 * (48*2) -> 27 * 27 * (48*2)
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 27 * 27 * (48*2) -> 27 * 27 * (128*2)
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            # 27 * 27 * (128*2) -> 13 * 13 * (128*2)
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 13 * 13 * (128*2) -> 13 * 13 * (192*2)
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            # 13 * 13 * (192*2) -> 13 * 13 * (192*2)
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            # 13 * 13 * (192*2) -> 13 * 13 * (128*2)
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            # 13 * 13 * (128*2) -> 6 * 6 * (128*2)
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(6 * 6 * 128 * 2, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU()
        )
        self.classifier = nn.Linear(2048, num_classes)
        if init_weights:
            self._initialize_weights()

    @autocast()
    def forward(self, x):
        x = self.layer(x)
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
