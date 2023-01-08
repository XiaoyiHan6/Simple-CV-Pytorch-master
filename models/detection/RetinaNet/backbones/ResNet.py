import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, \
    resnet101, resnet152


class ResNet(nn.Module):
    def __init__(self, resnet_type="resnet50", pretrained=False):
        super(ResNet, self).__init__()
        if resnet_type == "resnet18":
            self.model = resnet18(pretrained=pretrained)
        elif resnet_type == "resnet34":
            self.model = resnet34(pretrained=pretrained)
        elif resnet_type == "resnet50":
            self.model = resnet50(pretrained=pretrained)
        elif resnet_type == "resnet101":
            self.model = resnet101(pretrained=pretrained)
        elif resnet_type == "resnet152":
            self.model = resnet152(pretrained=pretrained)
        del self.model.fc
        del self.model.avgpool

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
    backbone = ResNet(resnet_type='resnet18', pretrained=True)
    x = torch.randn([16, 3, 512, 512])
    C3, C4, C5 = backbone(x)
    print(C3.shape)  # torch.Size([16, 512, 64, 64])
    print(C4.shape)  # torch.Size([16, 1024, 32, 32])
    print(C5.shape)  # torch.Size([16, 2048, 16, 16])
