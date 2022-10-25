import torch
import torch.nn as nn
from models.detection.RetinaNet.utils.iou import cal_ciou


class FocalLoss(nn.Module):
    # coco xywh -> xyxy
    def forward(self, cls_heads, reg_heads, anchors, annots):
        alpha = 0.25
        gamma = 2.0
        batch_size = cls_heads.shape[0]
        cls_losses = []
        reg_losses = []

        # (..., 4) 4 indicate location
        anchor = anchors[0, :, :]

        # anchor is xyxy, so change it to xywh
        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):
            # (batch_size, ?, num_classes)
            cls_head = cls_heads[j, :, :]
            # (batch_size, ?, 4)
            reg_head = reg_heads[j, :, :]

            # (batch_size, ?, 5)
            # (x, y, w, h, cls)
            bbox_annot = annots[j]
            # delete the bbox marked -1
            bbox_annot = bbox_annot[bbox_annot[:, 4] != -1]

            # limit (1e-4, 1.0 - 1e-4)
            cls_head = torch.clamp(cls_head, 1e-4, 1.0 - 1e-4)

            # if there is no bbox.
            if bbox_annot.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(cls_head.shape).cuda() * alpha
                else:
                    alpha_factor = torch.ones(cls_head.shape) * alpha
                # (1-a)
                alpha_factor = 1. - alpha_factor
                focal_weight = cls_head
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                bce = -(torch.log(1.0 - cls_head))
                cls_loss = focal_weight * bce
                cls_losses.append(cls_loss.sum())

                if torch.cuda.is_available():
                    reg_losses.append(torch.tensor(0).float().cuda())
                else:
                    reg_losses.append(torch.tensor(0).float())

                continue
            # num_anchors .vs. num_annotations
            IoU = cal_ciou(anchors[0, :, :], bbox_annot[:, :4])

            # max: value idx
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            # compute the loss for classification
            targets = torch.ones(cls_head.shape) * -1

            if torch.cuda.is_available():
                targets = targets.cuda()
            # set 0 background (IoU_max < 0.4)
            # (num_anchor, 1)
            targets[torch.lt(IoU_max, 0.4), :] = 0

            # set positive_indices (IoU_max > 0.5), return [f, t, f, f, t, ...]
            # (num_anchor, 1)
            positive_indices = torch.ge(IoU_max, 0.5)

            # count num
            num_positive_anchors = positive_indices.sum()

            # (num_anchor, 4)
            assigned_annots = bbox_annot[IoU_argmax, :]

            targets[positive_indices, :] = 0

            # pos examples label 1
            # assigned_annots[positive_indices, 4]: idxs -> True -> id
            # long(): delete the decimal point
            # idxs -> True  -> 1
            targets[positive_indices, assigned_annots[positive_indices, 4].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            # a = 0.25 if targets == 1 else 0.75
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - cls_head, cls_head)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(cls_head) + (1.0 - targets) * torch.log(1.0 - cls_head))

            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                # if != -1
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            cls_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression
            if positive_indices.sum() > 0:
                assigned_annots = assigned_annots[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                # gt xyxy -> xywh
                gt_widths = assigned_annots[:, 2] - assigned_annots[:, 0]
                gt_heights = assigned_annots[:, 3] - assigned_annots[:, 1]
                gt_ctr_x = assigned_annots[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annots[:, 1] + 0.5 * gt_heights

                # clip width to 1
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                # smooth L1 loss
                # gt (anchor) -> targets
                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                # concatenates a sequence of tensors along a new dimension.
                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                # negative_indices = 1 + (~positive_indices)

                # smooth L1 Loss
                reg_diff = torch.abs(targets - reg_head[positive_indices, :])

                # reg_loss = 0.5*9.0*torch.pow(reg_diff,2) if reg_diff < 1.0/9.0 else reg_diff - 0.5/9.0
                reg_loss = torch.where(
                    torch.le(reg_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(reg_diff, 2),
                    reg_diff - 0.5 / 9.0
                )
                reg_losses.append(reg_loss.mean())
            else:
                if torch.cuda.is_available():
                    reg_losses.append(torch.tensor(0).float().cuda())
                else:
                    reg_losses.append(torch.tensor(0).float())

        return torch.stack(cls_losses).mean(dim=0, keepdim=True), \
               torch.stack(reg_losses).mean(dim=0, keepdim=True)


if __name__ == '__main__':
    c = torch.rand([1, 4567, 80]).cuda()
    r = torch.randn([1, 4567, 4]).cuda()
    a = torch.randn([1, 4567, 4]).cuda()
    anno = torch.randn([1, 15, 5]).cuda()
    model = FocalLoss().cuda()
    out = model(c, r, a, anno)
    for i in range(len(out)):
        print(out[i])
