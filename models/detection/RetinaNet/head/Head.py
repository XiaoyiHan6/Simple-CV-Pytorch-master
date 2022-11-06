# clsHead and regHead for there are four times (3x3 conv+relu) in the first half
# then, for clsHead, add 3x3 conv whose channels are (num_anchors x num_classes)
# then, for regHead, add 3x3 conv whose channels are (num_anchors x 4)

# num_anchors = total anchor of all levels FPN feature maps
import torch
import torch.nn as nn
from torch.cuda.amp import autocast


class clsHead(nn.Module):
    def __init__(self,
                 inplanes,
                 num_anchors=9,
                 num_classes=80,
                 prior=0.01,
                 planes=256):
        super(clsHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.prior = prior
        self.cls_head = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(planes, num_anchors * num_classes, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid())

    @autocast()
    def forward(self, x):
        x = self.cls_head(x)
        # shape of x: (batch_size, C, H, W) with C = num_classes * num_anchors
        # shape of out: (batch_size, H, W, num_classes * num_anchors)
        out = x.permute(0, 2, 3, 1)
        # shape: (batch_size, H*W*num_anchors, num_classes)
        out = out.contiguous().view(out.shape[0], -1, self.num_classes)
        del x
        return out


class regHead(nn.Module):
    def __init__(self,
                 inplanes,
                 num_anchors=9,
                 planes=256):
        super(regHead, self).__init__()
        self.reg_head = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(inplanes, num_anchors * 4, kernel_size=3, stride=1, padding=1))

    @autocast()
    def forward(self, x):
        out = self.reg_head(x)
        # shape of x: (batch_size, C, H, W), with C = 4*num_anchors
        # shape of out: (batch_size, H, W, 4*num_anchors)
        out = out.permute(0, 2, 3, 1)

        # shape : (batch_size, H*W*num_anchors, 4)
        out = out.contiguous().view(out.shape[0], -1, 4)
        del x
        return out


if __name__ == "__main__":
    C = torch.randn([2, 256, 512, 512])
    ClsHead = clsHead(256)
    RegHead = regHead(256)
    out = RegHead(C)
    print(out.shape)
    for i in range(len(out)):
        print(out[i].shape)
    # torch.Size([2, 2359296, 4])
    # torch.Size([2359296, 4])
    # torch.Size([2359296, 4])

    C1 = torch.randn([2, 256, 64, 64])
    C2 = torch.randn([2, 256, 32, 32])
    C3 = torch.randn([2, 256, 16, 16])
    C4 = torch.randn([2, 256, 8, 8])
    C5 = torch.randn([2, 256, 4, 4])

    ClsHead = clsHead(256)
    print(ClsHead(C1).shape)  # torch.Size([2, 36864, 80])
    print(ClsHead(C2).shape)  # torch.Size([2, 9216, 80])
    print(ClsHead(C3).shape)  # torch.Size([2, 2304, 80])
    print(ClsHead(C4).shape)  # torch.Size([2, 576, 80])
    print(ClsHead(C5).shape)  # torch.Size([2, 144, 80])
