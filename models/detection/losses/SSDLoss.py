import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def hard_neg_mining(loss_c, labels, neg_pos_ratio):
    """
    Args:
    loss: (N, num_anchors): the loss for each example.
    labels: (N, num_anchors): the labels.
    neg_pos_ratio: the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio
    loss_c[pos_mask] = -math.inf
    _, index = loss_c.sort(dim=1, descending=True)
    _, idx_rank = loss_c.sort(dim=1)
    neg_mask = idx_rank < num_neg
    return pos_mask | neg_mask


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

    # match wh /anchor wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / anchors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]

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


def match(gt_truths, gt_labels, anchors, variances, threshold):
    """
    Args:
        gt_truths: (tensor) Ground truth boxes, Shape: [num_obj,4].
        gt_labels: (tensor) All the class labels for the image, Shape: [num_obj].
        anchors: (tensor) Anchor boxes from anchor layers, Shape: [num_anchors, 4].
        variances: (tensor) Variances corresponding to each anchor coord, Shape: [2].
        threshold: (float) The overlap threshold used when mathing boxes.
    Return:
        boxes: Shape: [num_anchors,4]
        labels: Shape: [num_anchors]
    """
    # overlap.shape: [num_obj,num_anchors]
    overlaps = jaccard(gt_truths, point_form(anchors))

    # size: num_anchors
    best_anchor_overlap, best_anchor_idx = overlaps.max(dim=1, keepdim=True)

    # size: num_obj
    best_truth_overlap, best_truth_idx = overlaps.max(dim=0, keepdim=True)

    best_truth_overlap.squeeze_(0)
    best_truth_idx.squeeze_(0)

    best_anchor_overlap.squeeze_(1)
    best_anchor_idx.squeeze_(1)

    best_truth_overlap.index_fill_(0, best_anchor_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_anchor_idx.size(0)):
        best_truth_idx[best_anchor_idx[j]] = j
    labels = gt_labels[best_truth_idx] + 1
    labels[best_truth_overlap < threshold] = 0
    boxes = gt_truths[best_truth_idx]
    location = encode(boxes, anchors, variances)
    return location, labels


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
        # total_anchor_nums=8732, num_classes=21
        # loc_data.shape: [batch_size, total_anchor_nums, 4]
        # conf_data.shape: [batch_size, total_anchor_nums, num_classes]
        # anchors.shape: [total_anchor_nums, 4]
        loc_data, conf_data, anchors = predictions
        device = anchors.device

        batch_size = loc_data.shape[0]
        num_anchors = loc_data.shape[1]

        # anchor boxes
        loc_t = torch.Tensor(batch_size, num_anchors, 4).to(device)
        # anchor labels
        conf_t = torch.LongTensor(batch_size, num_anchors).to(device)

        for i in range(batch_size):
            # targets.shape: [batch_size, num_obj, 5]
            # truths.shape: [num_obj, 4]
            # labels.shape:[num_obj]
            gt_truths = targets[i][:, :-1].data
            gt_labels = targets[i][:, -1].data

            location, labels = match(gt_truths, gt_labels, anchors, variances=self.variances,
                                     threshold=self.threshold)

            loc_t[i, :, :] = location
            conf_t[i, :] = labels
        num_classes = conf_data.size(2)
        with torch.no_grad():
            loss_c = -F.log_softmax(conf_data, dim=2)[:, :, 0]
            mask = hard_neg_mining(loss_c, conf_t, self.negpos_ratio)
        conf_data = conf_data[mask, :]
        loss_c = F.cross_entropy(conf_data.view(-1, num_classes), conf_t[mask], reduction='sum')

        pos_mask = conf_t > 0
        loc_t = loc_t[pos_mask, :].view(-1, 4)
        loc_data = loc_data[pos_mask, :].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_data, loc_t, reduction='sum')

        num_pos = loc_t.size(0)

        loss_c /= num_pos
        loss_l /= num_pos

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
    label = torch.randint(20, (1, 10, 1))
    t = torch.cat((loc, label.float()), dim=2)

    c, l = loss(p, t)
    print("conf_loss: ", c, ", loc_loss: ", l)
