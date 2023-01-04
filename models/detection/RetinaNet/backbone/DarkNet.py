import torch.nn as nn
from torchvision import models


class Darknet19(nn.Module):
    def __init__(self):
        super(Darknet19, self).__init__()
        # self.model = models.__dict__['darknet19'](**{"pretrained": True})
        del self.model.avgpool
        del self.model.layer7

    def forward(self, x):
        x = self.model.layer1(x)
        x = self.model.maxpool(x)
        x = self.model.layer2(x)
        C3 = self.model.layer3(x)
        C4 = self.model.layer4(C3)
        C5 = self.model.layer5(C4)
        C5 = self.model.layer6(C5)

        del x
        return [C3, C4, C5]


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        # self.model = models.__dict__['darknet53'](**{"pretrained": True})
        del self.model.fc
        del self.model.avgpool

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.conv2(x)
        x = self.model.block1(x)
        x = self.model.conv3(x)
        x = self.model.block2(x)
        x = self.model.conv4(x)
        C3 = self.model.block3(x)
        C4 = self.model.conv5(C3)
        C4 = self.model.block4(C4)
        C5 = self.model.conv6(C4)
        C5 = self.model.block5(C5)
        del x
        return [C3, C4, C5]
