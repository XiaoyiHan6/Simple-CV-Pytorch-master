import numpy as np
import torch
from torch import nn
from utils.nms import ssd_nms
import torch.nn.functional as F
from torch.cuda.amp import autocast
from models.detection.anchors.SSDAnchors import SSDAnchors
from models.detection.losses.SSDLoss import SSDLoss, decode
from models.detection.heads.SSDHeads import confHeads, locHeads
from models.detection.backbones.VggNetBackbone import VggNetBackbone


class SSD(nn.Module):
    def __init__(self, size=300, backbones=VggNetBackbone,
                 loc_heads=locHeads, conf_heads=confHeads,
                 version='VOC', training=False, batch_norm=False):
        super(SSD, self).__init__()
        """
        :param
        phase: test/train
        size: image size
        backbones: VggNetBackbone
        necks: SSDNecks
        heads: SSDHeads
        dataset: dataset type
        num_classes: number of classes
        """
        self.training = training
        self.size = size
        self.version = version
        if version == 'VOC':
            self.num_classes = 21

        elif version == 'COCO':
            self.num_classes = 81
        else:
            raise ValueError("Dataset type is error!")

        # backbone
        self.backbone = backbones(batch_norm=batch_norm)
        # head
        self.loc_heads = loc_heads(self.backbone, batch_norm=batch_norm)
        self.conf_heads = conf_heads(self.backbone, num_classes=self.num_classes, batch_norm=batch_norm)
        self.anchors = SSDAnchors(version=self.version)
        self.freeze_bn()
        # training
        self.loss = SSDLoss(num_classes=self.num_classes)
        # inference
        self.top_k = 200
        self.conf_thresh = 0.01
        self.nms_thresh = 0.5
        self.variance = [0.1, 0.2]

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @autocast()
    def forward(self, inputs):
        if self.training:
            img_batch, annots = inputs
        else:
            img_batch = inputs
        device = img_batch.device

        features = self.backbone(img_batch)

        # (batch_size, total_anchor_nums, 4)
        loc_heads = self.loc_heads(features)

        # (batch_size, total_anchor_nums, num_clesses)
        conf_heads = self.conf_heads(features)

        del features

        anchors = self.anchors()
        anchors = anchors.to(device)
        if self.training:
            return self.loss([loc_heads, conf_heads, anchors], annots)
        else:
            # conf_hedas.shape: [batch_size,num_anchors,num_classes] -> [16,8732,21]
            conf_heads = F.softmax(conf_heads, dim=2)
            # loc_heads.shape: [batch_size,num_anchors,4] -> [16,8732,4]
            # anchors.shape [num_anchors,4] -> [8732,4]
            batch_size = loc_heads.size(0)
            num_anchors = anchors.size(0)
            device = anchors.device
            # decode_boxes.shape: [batch_size,num_anchors ,4]
            decode_boxes = torch.Tensor(batch_size, num_anchors, 4).to(device)
            for i in range(batch_size):
                decode_boxes_i = decode(loc_heads[i], anchors, self.variance)
                decode_boxes[i] = decode_boxes_i
            # detections = (conf_heads, decode_boxes)
            results = []
            for i in range(batch_size):
                # scores.shape: [num_anchors, num_classes]
                # boxes.shape: [num_anchors, 4]
                scores, boxes = conf_heads[i], decode_boxes[i]
                # boxes.shape: [num_anchors, num_classes,4]
                boxes = boxes.view(num_anchors, 1, 4).expand(num_anchors, self.num_classes, 4)
                labels = torch.arange(self.num_classes, device=device)
                # labels.shape: [scores.size(0), num_classes] -> [num_anchors, num_classes]
                labels = labels.view(1, self.num_classes).expand_as(scores)
                # remove background label
                boxes = boxes[:, 1:]
                scores = scores[:, 1:]
                labels = labels[:, 1:]
                # batch everything, by making every class prediction be a separate instance
                # boxes.shape: [num_classes*num_anchors, 4] -> [20*8732, 4]
                # scores.shape and labels.shape: [num_classes*num_anchors]
                boxes = boxes.reshape(-1, 4)
                scores = scores.reshape(-1)
                labels = labels.reshape(-1)

                # remove low scoring boxes
                indices = torch.nonzero(scores > self.conf_thresh).squeeze(1)
                boxes, scores, labels = boxes[indices], scores[indices], labels[indices]

                boxes[:, 0::2] *= 300
                boxes[:, 1::2] *= 300
                keep = ssd_nms(boxes, scores, labels, self.nms_thresh)
                keep = keep[:self.top_k]
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                results.append([boxes, scores, labels])
            return results


if __name__ == "__main__":
    ssd1 = SSD(training=False)
    x = torch.randn(1, 3, 300, 300)
    loc = torch.randn(1, 10, 4)

    label = torch.randint(20, (1, 10, 1))
    annot = torch.cat((loc, label.float()), dim=2)

    input = x
    output = ssd1(input)
    for i in range(len(output)):
        boxes, scores, labels = output[i]
        print(i, ", boxes: ", boxes, "\nscores:", scores, "\nlabels", labels)

    ssd2 = SSD(training=True)
    input = x, annot
    loss_c, loss_l = ssd2(input)
    print("loss_c:", loss_c.data, " ,loss_l:", loss_l.data)
