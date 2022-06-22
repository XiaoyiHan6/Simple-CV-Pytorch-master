from torch import nn


class lenet5(nn.Module):
    def __init__(self):
        super(lenet5, self).__init__()
        self.model = nn.Sequential(
            # input:3@32x32
            # 6@28x28
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=0, stride=1),
            nn.ReLU(inplace=True),
            # 6@14x14
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            #  16@10x10
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0, stride=1),
            nn.ReLU(inplace=True),
            # 16@5x5
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            # use cifar dataset
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x
