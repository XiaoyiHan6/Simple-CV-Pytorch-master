import torch.nn as nn
from models.detection import backbones


class Darknet19Backbone(nn.Module):
    def __init__(self):
        super(Darknet19Backbone, self).__init__()
        self.model = backbones.__dict__['darknet19'](**{"pretrained": True})
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

