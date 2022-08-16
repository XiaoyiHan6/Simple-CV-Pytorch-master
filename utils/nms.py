from torch import Tensor
import torch
from utils.iou import cal_ciou


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed.
        They are expected to be in (x1, y1, x2, y2) format.

    Returns:
       area(Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def nms(boxes, scores, overlap=0.5):
    """Arguments:
        boxes:[N,4]
        scores:[N]
        iou_threshold:0.5
    Returns:
    """
    keep = []
    idxs = scores.argsort()  # from low to high
    while idxs.numel() > 0:
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]  # [1,4]
        keep.append(max_score_index)
        if idxs.size(0) == 1:
            break
        idxs = idxs[:-1]
        other_boxes = boxes[idxs]
        ious = cal_ciou(max_score_box, other_boxes)
        idxs = idxs[ious[0] <= overlap]

    keep = idxs.new(keep)  # Tensor
    return keep


def SSDNMS(boxes, scores, threshold=0.5, top_k=200):
    """
    boxes: pred boxes, Shape: [M,4]
    scores: conf, Shape: [M]
    threshold:
    top_k: top k boxes
    :return:
    keep: boxes after nms -> index
    count: boxes
    """
    keep = scores.new(scores.size(0)).zero_().long()
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1) * (y2 - y1)
    _, idx = scores.sort(0, descending=True)
    idx = idx[:top_k]
    count = 0
    while idx.numel():
        i = idx[0]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[1:]
        xx1 = x1[idx].clamp(min=x1[i])
        yy1 = y1[idx].clamp(min=y1[i])
        xx2 = x2[idx].clamp(max=y2[i])
        yy2 = y2[idx].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        iou = inter / (area[i] + area[idx] - inter)
        idx = idx[iou.le(threshold)]
    return keep, count
