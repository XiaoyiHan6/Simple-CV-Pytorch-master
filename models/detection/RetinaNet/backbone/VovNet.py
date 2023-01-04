from torch import nn


class VovNet(nn.Module):
    def __init__(self, vovnet_type='VoVNet39_se'):
        super(VovNet, self).__init__()
        # self.model = models.__dict__[vovnet_type](**{"pretrained": True})
        del self.model.fc
        del self.model.avgpool

    def forward(self, x):
        x = self.model.stem(x)
        features = []
        for stage in self.model.stages:
            x = stage(x)
            features.append(x)
        del x
        return features[1:]
