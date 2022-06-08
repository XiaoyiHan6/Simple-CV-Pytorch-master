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
