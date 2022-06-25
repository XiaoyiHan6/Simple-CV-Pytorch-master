from torch import nn
from torch.cuda.amp import autocast


class lenet5(nn.Module):
    # cifar: 10, ImageNet: 1000
    def __init__(self, num_classes=1000, init_weights=False):
        super(lenet5, self).__init__()
        self.num_classes = num_classes
        self.layers = nn.Sequential(
            # input:32 * 32 * 3 -> 28 * 28 * 6
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=0, stride=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            # 28 * 28 * 6 -> 14 * 14 * 6
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # 14 * 14 * 6 -> 10 * 10 * 16
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 10 * 10 * 16 -> 5 * 5 * 16
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84))
        self.classifier = nn.Linear(84, self.num_classes)

        if init_weights:
            self._initialize_weights()

    @autocast()
    def forward(self, x):
        x = self.layers(x)
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
