import torch
import torch.nn as nn
import torch.nn.functional as F


def encode(matched, anchors, variances):
    """

    matched: (tensor) Coords of ground truth for each anchor in point-form, Shape: [num_anchors,4].
    anchors: (tensor) Anchor boxes in center-offset form, Shape: [num_anchors,4].
    variances: (list[float]) Variances of anchor boxes
    :return:
    encoded boxes (tensor), Shape:[num_anchors,4]
    """
    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - anchors[:, :2]

    # encode variance
    # shape [num_anchors,2]
    g_cxcy /= (anchors[:, 2:] * variances[0])

    eps = 1e-5
    # match wh /anchor wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / anchors[:, 2:]
    g_wh = torch.log(g_wh + eps) / variances[1]

    # return target for smooth_L1_loss
    # [num_anchors,4]
    return torch.cat([g_cxcy, g_wh], 1)


def decode(loc, anchors, variances):
    boxes = torch.cat((anchors[:, :2] + loc[:, :2] * variances[0] * anchors[:, 2:],
                       anchors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)

    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def point_form(boxes):
    """
    (cx,cy,w,h) -> (xmin,ymin,xmax,ymax)
    boxes: (tensor) center-size default boxes from anchor layers.
    :return:
    (tensor) Converted xmin,ymin,xmax,ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)


def center_size(boxes):
    """
    (xmin,ymin,xmax,ymax) -> (cx,cy,w,h)
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2]) / 2,
                     (boxes[:, 2:] - boxes[:, :2]), 1)


def intersect(box_a, box_b):
    """
     We resize both tensors to [A, B, 2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    box_a: (tensor) bounding boxes, Shape: [A, 4].
    box_b: (tensor) bounding boxes, Shape: [B, 4].
    :return:
    (tensor) intersection area, Shape: [A, B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    # RB
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    # LT
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """
     A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    box_a: (tensor) Ground truth bounding boxes, Shape: [num_obj, 4]
    box_b: (tensor) Anchor boxes from anchor layers, Shape: [num_obj, 4]
    :return:
    jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def match(threshold, truths, anchors, variances, labels, loc_t, conf_t, idx):
    """
    threshold: (float) The overlap threshold used when mathing boxes.
    truths: (tensor) Ground truth boxes, Shape: [num_obj, num_anchors].
    anchors: (tensor) Anchor boxes from anchor layers, Shape: [num_anchors, 4].
    variances: (tensor) Variances corresponding to each anchor coord, Shape: [2]
    labels: (tensor) All the class labels for the image, Shape: [num_obj].
    loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
    conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
    idx: (int) current batch index

    :return
    The matched indices corresponding to 1) location and 2) confidence preds.
    """
    # jaccard index
    overlaps = jaccard(truths, point_form(anchors))

    # (Bipartite, Matching)
    # [1, num_obj] best anchor for each ground truth
    best_anchor_overlap, best_anchor_idx = overlaps.max(1, keepdim=True)

    # [1, num_anchors] best ground truth for each anchor
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)

    best_anchor_idx.squeeze_(1)
    best_anchor_overlap.squeeze_(1)

    # ensure best anchor
    # 2 > threshold
    best_truth_overlap.index_fill_(0, best_anchor_idx, 2)

    # ensure every gt matches with its anchor of max overlap
    for j in range(best_anchor_idx.size(0)):
        best_truth_idx[best_anchor_idx[j]] = j

    # Shape: [num_anchors,4]
    matches = truths[best_truth_idx]

    # Shape: [num_anchors]
    conf = labels[best_truth_idx] + 1
    # label as background
    # bg 0
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, anchors, variances)

    # [num_anchors,4]
    # encoded offsets to learn
    loc_t[idx] = loc

    # [num_anchors]
    # top class label for each anchor
    conf_t[idx] = conf


def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


class SSDLoss(nn.Module):
    """
        P:F 1:3
        L(x, c, l, g) = (Lconf(x, c) + αLloc(x, l, g)) /N
        c: class confidences,
        l: predicted boxes,
        g: ground truth boxes,
        N: number of matched default boxes
    """

    def __init__(self, overlap_thresh=0.5,
                 anchor_for_matching=True,
                 bkg_label=0,
                 neg_mining=True,
                 neg_pos=3,
                 neg_overlap=0.5,
                 encode_target=False,
                 variances=[0.1, 0.2],
                 num_classes=21):
        super(SSDLoss, self).__init__()

        self.num_classes = num_classes

        # IoU 0.5
        self.threshold = overlap_thresh
        # background label 0
        self.background_label = bkg_label
        # False
        self.encode_target = encode_target
        # True
        self.use_anchor_for_matching = anchor_for_matching
        # True
        self.do_neg_mining = neg_mining
        # F:P 3:1
        self.negpos_ratio = neg_pos
        # 0.5
        self.neg_overlap = neg_overlap
        self.num_classes = num_classes
        self.variances = variances

    def forward(self, predictions, targets):
        # total_anchor_nums=8732, num_classes=20
        # loc_data.shape: [batch_size, total_anchor_nums, 4]
        # conf_data.shape: [batch_size, total_anchor_nums, num_classes]
        # anchors.shape: [total_anchor_nums, 4]
        loc_data, conf_data, anchors = predictions
        device = anchors.device

        # batch_size
        batch_size = loc_data.size(0)

        # loc_data.size(1) = 8732
        anchors = anchors[:loc_data.size(1), :]

        # num_anchors = 8732
        num_anchors = (anchors.size(0))

        # anchors & ground truth boxes
        # loc_t.shape: [batch_size,    8732,   4]
        loc_t = torch.Tensor(batch_size, num_anchors, 4).to(device)

        # conf_t.shape: [batch_size,  8732]
        conf_t = torch.LongTensor(batch_size, num_anchors).to(device)
        for idx in range(batch_size):
            # targets.shape: [num_objs,5]
            # truths.shape: [num_objs,4]
            truths = targets[idx][:, :-1].data

            # labels.shape; [num_objs]
            labels = targets[idx][:, -1].data

            # anchors.data.shape:[8732, 4]
            match(self.threshold, truths, anchors.data, self.variances, labels, loc_t, conf_t, idx)

        pos = conf_t > 0
        # Localization Loss (Smooth L1)
        # loc_data.shape: [batch_size,num_anchors,4]
        # pos.shape: [batch_size,num_anchors]
        # pos_idx.shape: [batch_size,num_anchors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        # gt
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        # conf_data.shape: [batch_size,num_anchors,num_classes]
        # batch_conf.data:[batch_size*num_anchors,num_classes]
        # reshape

        # conf_t.shape: [batch_size,num_anchors]
        # loss_c.shape: [batch_size*num_anchors,1]
        loss_c = -F.log_softmax(conf_data, dim=2)[:, :, 0]
        # Hard Negative Mining
        # filter out pos boxes for now
        # loss_c.shape: [batch_size*num_anchors,1]
        loss_c = loss_c.view(batch_size, -1)

        loss_c[pos] = 0
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        # num_pos.shape: [batch_size,1]
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # Confidence Loss Including Positive and Negative Examples
        # pos.shape: [batch_size,num_anchors]
        # pos_idx.shape: [batch_size,num_anchors,num_classes]
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        # neg.shape: [batch_size,num_anchors]
        # neg_idx.shape: [batch_size,num_anchors,num_classes]
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        # pos_idx,neg_idx
        # conf_data.shape:[batch_size,num_anchors,num_classes]
        # conf_p.shape: [num_obj,num_classes]
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        # conf_t.shape:[batch_size,num_anchors]
        # targets_weighted.shape:[num_obj]
        targets_weighted = conf_t[(pos + neg).gt(0)]

        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x,c)+ αLloc(x,l,g))/N
        # N = num_pos.data.sum().float()
        loss_c /= pos_idx.sum()
        loss_l /= pos_idx.sum()
        return loss_c, loss_l


if __name__ == "__main__":
    # IoU
    box_a = torch.Tensor([[2, 1, 4, 3]])
    box_b = torch.Tensor([[3, 2, 5, 4]])
    print("IoU:", jaccard(box_a, box_b))

    # Loss
    loss = SSDLoss()
    l = torch.randn(1, 100, 4)
    c = torch.randn(1, 100, 21)
    anchors = torch.randn(100, 4)
    p = (l, c, anchors)

    loc = torch.randn(1, 10, 4)
    label = torch.randint(21, (1, 10, 1))
    t = torch.cat((loc, label.float()), dim=2)

    c, l = loss(p, t)
    print("conf: ", c, ", loss: ", l)
