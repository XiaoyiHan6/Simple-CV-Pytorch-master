import torch
from torch import nn
from models.classification.resnet import resnet18, resnet34, resnet50, \
    resnet101, resnet152


class ResNetBackbone(nn.Module):
    def __init__(self, resnet_type="resnet50", pretrained=False, include_top=False):
        super(ResNetBackbone, self).__init__()
        if resnet_type == "resnet18":
            self.model = resnet18(pretrained=pretrained, include_top=include_top)
        elif resnet_type == "resnet34":
            self.model = resnet34(pretrained=pretrained, include_top=include_top)
        elif resnet_type == "resnet50":
            self.model = resnet50(pretrained=pretrained, include_top=include_top)
        elif resnet_type == "resnet101":
            self.model = resnet101(pretrained=pretrained, include_top=include_top)
        elif resnet_type == "resnet152":
            self.model = resnet152(pretrained=pretrained, include_top=include_top)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        C3 = self.model.layer2(x)
        C4 = self.model.layer3(C3)
        C5 = self.model.layer4(C4)

        del x

        return [C3, C4, C5]


if __name__ == "__main__":
    backbone = ResNetBackbone(pretrained=True)
    x = torch.randn([4, 3, 512, 512])
    C3, C4, C5 = backbone(x)
    print(C3.shape)
    print(C4.shape)
    print(C5.shape)
