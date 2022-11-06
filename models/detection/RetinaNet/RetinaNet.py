import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import math
import torch
import torch.nn as nn
from torchvision.ops import nms
from utils.path import CheckPoints
from torch.cuda.amp import autocast
from models.detection.RetinaNet.neck import FPN
from models.detection.RetinaNet.head import clsHead, regHead
from models.detection.RetinaNet.anchor import RetinaNetAnchors
from models.detection.RetinaNet.backbone import ResNetBackbone
from models.detection.RetinaNet.utils.ClipBoxes import ClipBoxes
from models.detection.RetinaNet.loss.Loss import FocalLoss
from models.detection.RetinaNet.utils.BBoxTransform import BBoxTransform

__all__ = [
    'resnet18_retinanet',
    'resnet34_retinanet',
    'resnet50_retinanet',
    'resnet101_retinanet',
    'resnet152_retinanet',
]

model_urls = {
    'resnet18_retinanet':
        '{}/resnet18-5c106cde.pth'.format(CheckPoints),
    'resnet34_retinanet':
        '{}/resnet34-333f7ec4.pth'.format(CheckPoints),
    'resnet50_retinanet':
        '{}/resnet50-19c8e357.pth'.format(CheckPoints),
    'resnet101_retinanet':
        '{}/resnet101-5d3b4d8f.pth'.format(CheckPoints),
    'resnet152_retinanet':
        '{}/resnet152-b121ed2d.pth'.format(CheckPoints),
}


# assert input annotations are [x_min, y_min, x_max, y_max]
class RetinaNet(nn.Module):
    def __init__(self,
                 resnet_type,
                 num_classes=80,
                 planes=256,
                 pretrained=False,
                 training=False):
        super(RetinaNet, self).__init__()
        self.resnet_type = resnet_type,
        # coco 80, voc 20
        self.num_classes = num_classes
        self.planes = planes
        self.training = training
        self.backbone = ResNetBackbone(resnet_type=resnet_type,
                                       pretrained=pretrained)
        expand_ratio = {
            "resnet18": 1,
            "resnet34": 1,
            "resnet50": 4,
            "resnet101": 4,
            "resnet152": 4
        }

        C3_inplanes, C4_inplanes, C5_inplanes = \
            int(128 * expand_ratio[resnet_type]), \
            int(256 * expand_ratio[resnet_type]), \
            int(512 * expand_ratio[resnet_type])

        self.fpn = FPN(C3_inplanes=C3_inplanes,
                       C4_inplanes=C4_inplanes,
                       C5_inplanes=C5_inplanes,
                       planes=self.planes)

        self.cls_head = clsHead(inplanes=self.planes,
                                num_classes=self.num_classes)

        self.reg_head = regHead(inplanes=self.planes)

        self.anchors = RetinaNetAnchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

        self.loss = FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = self.cls_head.prior
        b = -math.log((1.0 - prior) / prior)
        self.cls_head.cls_head[-2].weight.data.fill_(0)
        self.cls_head.cls_head[-2].bias.data.fill_(b)

        self.reg_head.reg_head[-1].weight.data.fill_(0)
        self.reg_head.reg_head[-1].bias.data.fill_(0)

        self.freeze_bn()

    def freeze_bn(self):
        """
        Freeze BatchNorm layers.
        """
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    @autocast()
    def forward(self, inputs):
        if self.training:
            img_batch, annots = inputs

        # inference
        else:
            img_batch = inputs

        [C3, C4, C5] = self.backbone(img_batch)

        del inputs

        features = self.fpn([C3, C4, C5])
        del C3, C4, C5

        # (batch_size, total_anchors_nums, num_classes)
        cls_heads = torch.cat([self.cls_head(feature) for feature in features], dim=1)

        # (batch_size, total_anchors_nums, 4)
        reg_heads = torch.cat([self.reg_head(feature) for feature in features], dim=1)

        del features

        anchors = self.anchors(img_batch)

        if self.training:
            return self.loss(cls_heads, reg_heads, anchors, annots)
        # inference
        else:
            transformed_anchors = self.regressBoxes(anchors, reg_heads)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            # scores
            finalScores = torch.Tensor([])

            # anchor id:0~79
            finalAnchorBoxesIndexes = torch.Tensor([]).long()

            # coordinates size:[...,4]
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            # num_classes
            for i in range(cls_heads.shape[2]):
                scores = torch.squeeze(cls_heads[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue
                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                # use idx to find the scores of anchor
                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                # [0,0,0,...,1,1,1,...,79,79]
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])

                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                # [...,4]
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

        return finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates


def _retinanet(num_classes, arch, pretrained, training, **kwargs):
    model = RetinaNet(num_classes=num_classes, resnet_type=arch, training=training, **kwargs)
    # only load state_dict()
    if pretrained:
        model.load_state_dict(torch.load(model_urls[arch + "_retinanet"]), strict=False)

    return model


def resnet18_retinanet(num_classes, pretrained=False, training=False, **kwargs):
    return _retinanet(num_classes, 'resnet18', pretrained, training, **kwargs)


def resnet34_retinanet(num_classes, pretrained=False, training=False, **kwargs):
    return _retinanet(num_classes, 'resnet34', pretrained, training, **kwargs)


def resnet50_retinanet(num_classes, pretrained=False, training=False, **kwargs):
    return _retinanet(num_classes, 'resnet50', pretrained, training, **kwargs)


def resnet101_retinanet(num_classes, pretrained=False, training=False, **kwargs):
    return _retinanet(num_classes, 'resnet101', pretrained, training, **kwargs)


def resnet152_retinanet(num_classes, pretrained=False, training=False, **kwargs):
    return _retinanet(num_classes, 'resnet152', pretrained, training, **kwargs)


if __name__ == "__main__":
    C = torch.randn([2, 3, 512, 512])
    model = RetinaNet(resnet_type="resnet50", num_classes=80, training=False)
    model = model.cuda()
    C = C.cuda()
    model = torch.nn.DataParallel(model).cuda()
    out = model(C)
    for i in range(len(out)):
        print(out[i].shape)
        print(out[i])

# Scores: torch.Size([486449])
# tensor([4.1057, 4.0902, 4.0597,  ..., 0.0509, 0.0507, 0.0507], device='cuda:0')
# Id: torch.Size([486449])
# tensor([ 0,  0,  0,  ..., 79, 79, 79], device='cuda:0')
# loc: torch.Size([486449, 4])
# tensor([[ 45.1607, 249.4807, 170.5788, 322.8085],
# [ 85.9825, 324.4150, 122.9968, 382.6297],
# [148.1854, 274.0474, 179.0922, 343.4529],
# ...,
# [222.5421,   0.0000, 256.3059,  15.5591],
# [143.3349, 204.4784, 170.2395, 228.6654],
# [208.4509, 140.1983, 288.0962, 165.8708]], device='cuda:0')
