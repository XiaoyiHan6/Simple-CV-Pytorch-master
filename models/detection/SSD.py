import torch
from torch import nn
from utils.nms import SSDNMS
from torch.cuda.amp import autocast
from models.detection.losses.SSDLoss import decode
from models.detection.losses.SSDLoss import SSDLoss
from models.detection.necks.SSDNecks import SSDNecks
from models.detection.anchors.SSDAnchors import SSDAnchors
from models.detection.heads.SSDHeads import confHeads, locHeads
from models.detection.backbones.VggNetBackbone import VggNetBackbone

backbone = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    '512': [],
}
neck = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [], }
head = {
    '300': [4, 6, 6, 6, 4, 4],
    '512': [],
}


class SSD(nn.Module):
    def __init__(self, size=300, backbones=VggNetBackbone,
                 necks=SSDNecks, loc_heads=locHeads, conf_heads=confHeads,
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
        self.backbone = backbones(cfg=backbone['300'], i=3, batch_norm=batch_norm)
        # neck
        self.neck = necks(cfg=neck['300'], i=1024, batch_norm=batch_norm)
        # head
        self.loc_heads = loc_heads(self.backbone, self.neck, cfg=head['300'])
        self.conf_heads = conf_heads(self.backbone, self.neck, cfg=head['300'], num_classes=self.num_classes)

        self.anchors = SSDAnchors(version=self.version)
        self.freeze_bn()
        if self.training:
            self.loss = SSDLoss(num_classes=self.num_classes)
        else:
            self.softmax = nn.Softmax(dim=-1)
            self.top_k = 200
            self.conf_thresh = 0.01
            self.nms_thresh = 0.45
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

        out1 = self.backbone(img_batch)
        out2 = self.neck(out1[1])

        features = [out1[0], out1[1], out2[0], out2[1], out2[2], out2[3]]
        del out1, out2

        # (batch_size, 4, num_clesses)
        # (batch_size,4*total_anchor_nums)
        loc_heads = self.loc_heads(features)

        # (batch_size, total_anchor_nums, num_clesses)
        # (batch_size,num_classes*total_anchor_nums)
        conf_heads = self.conf_heads(features)

        del features

        anchors = self.anchors(img_batch)
        # anchors = self.anchors()
        anchors = anchors.to(device)
        if self.training:
            return self.loss([loc_heads, conf_heads, anchors], annots)
        else:
            conf_heads = self.softmax(conf_heads)

            # loc_heads.shape [batch_size,num_anchors,4] -> [16,8732,4]
            # conf_hedas.shape [batch_size,num_anchors,num_classes] -> [16,8732,4]
            # anchors.shape [num_anchors,4] -> [8732,4]
            batch_size = loc_heads.size(0)
            # shape: [batch_size,num_classes,k,5]
            output = torch.zeros(batch_size, self.num_classes, self.top_k, 5)
            # conf_preds.shape [batch_size,4,8732]
            conf_preds = conf_heads.transpose(2, 1)

            for i in range(batch_size):
                decode_boxes = decode(loc_heads[i], anchors, self.variance)
                conf_scores = conf_preds[i].clone()
                for num in range(1, self.num_classes):
                    # mask conf < conf_thresh
                    c_mask = conf_scores[num].gt(self.conf_thresh)
                    scores = conf_scores[num][c_mask]
                    if scores.size(0) == 0:
                        continue
                    # conf < conf_thresh
                    l_mask = c_mask.unsqueeze(1).expand_as(decode_boxes)
                    boxes = decode_boxes[l_mask].view(-1, 4)
                    ids, count = SSDNMS(boxes, scores, self.nms_thresh, self.top_k)

                    pred_score = torch.zeros(count)
                    for id, x in enumerate(scores[ids[:count]]):
                        pred_score[id] = x

                    pred_label = torch.zeros(count)
                    for id in range(count):
                        pred_label[id] = num

                    pred_bbox = torch.zeros(count, 4)
                    for id, x in enumerate(boxes[ids[:count]]):
                        pred_bbox[id, :] = x
            return pred_score, pred_label, pred_bbox


if __name__ == "__main__":
    ssd1 = SSD(training=False)
    x = torch.randn(1, 3, 300, 300)
    loc = torch.randn(1, 10, 4)
    label = torch.randint(20, (1, 10, 1))
    annot = torch.cat((loc, label.float()), dim=2)

    input = x
    pred_score, pred_label, pred_bbox = ssd1(input)
    print("pred_score.shape:", pred_score.shape, ", pred_label:", pred_label, ", pred_bbox:", pred_bbox)

    ssd2 = SSD(training=True)
    input = x, annot
    loss_c, loss_l = ssd2(input)
    print("loss_c:", loss_c.data, ", loss_l:", loss_l.data)
