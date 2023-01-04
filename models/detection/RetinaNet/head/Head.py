# clsHead and regHead for there are four times (3x3 conv+relu) in the first half
# then, for clsHead, add 3x3 conv whose channels are (num_anchors x num_classes)
# then, for regHead, add 3x3 conv whose channels are (num_anchors x 4)

# num_anchors = total anchors of all levels FPN feature maps
import math
import torch
import torch.nn as nn


class clsHead(nn.Module):
    def __init__(self,
                 inplanes,
                 num_anchors=9,
                 num_classes=80,
                 planes=256):
        super(clsHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.output = nn.Conv2d(planes, num_anchors * num_classes, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)
        prior = 0.01
        b = -math.log((1 - prior) / prior)
        self.output.bias.data.fill_(b)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)

        out = self.conv2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.act(out)

        out = self.conv4(out)
        out = self.act(out)

        out = self.output(out)
        out = self.sigmoid(out)

        # shape of x: (batch_size, C, H, W) with C = num_classes * num_anchors
        # shape of out: (batch_size, H, W, num_classes * num_anchors)
        out = out.permute(0, 2, 3, 1)
        b, h, w, c = out.shape
        out = out.view(b, h, w, self.num_anchors, self.num_classes)
        # shape: (batch_size, H*W*num_anchors, num_classes)
        out = out.contiguous().view(x.shape[0], -1, self.num_classes)
        del x
        return out


class regHead(nn.Module):
    def __init__(self,
                 inplanes,
                 num_anchors=9,
                 planes=256):
        super(regHead, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.output = nn.Conv2d(inplanes, num_anchors * 4, kernel_size=3, padding=1)
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)
        self.output.weight.data.fill_(0)
        self.output.bias.data.fill_(0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)

        out = self.conv2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.act(out)

        out = self.conv4(out)
        out = self.act(out)

        out = self.output(out)
        # shape of x: (batch_size, C, H, W), with C = 4*num_anchors
        # shape of out: (batch_size, H, W, 4*num_anchors)
        out = out.permute(0, 2, 3, 1)

        # shape : (batch_size, H*W*num_anchors, 4)
        out = out.contiguous().view(out.shape[0], -1, 4)
        del x
        return out


if __name__ == "__main__":
    # B,C,H,W
    C = torch.randn([2, 256, 512, 512])
    RegHead = regHead(256)
    out = RegHead(C)
    print("RegHead out.shape:")
    print(out.shape)
    # torch.Size([2, 2359296, 4])

    print("********************************")

    C1 = torch.randn([2, 256, 64, 64])
    C2 = torch.randn([2, 256, 32, 32])
    C3 = torch.randn([2, 256, 16, 16])
    C4 = torch.randn([2, 256, 8, 8])
    C5 = torch.randn([2, 256, 4, 4])

    print("ClsHead out.shape:")
    ClsHead = clsHead(256)
    print(ClsHead(C1).shape)  # torch.Size([2, 36864, 80])
    print(ClsHead(C2).shape)  # torch.Size([2, 9216, 80])
    print(ClsHead(C3).shape)  # torch.Size([2, 2304, 80])
    print(ClsHead(C4).shape)  # torch.Size([2, 576, 80])
    print(ClsHead(C5).shape)  # torch.Size([2, 144, 80])
