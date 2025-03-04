import numpy as np
from numpy import ndarray
import torch
from torchvision.ops.boxes import box_area


def to_corners(bboxes):
    if type(bboxes) == ndarray:
        bboxes = torch.tensor(bboxes)
    x, y, w, h = bboxes.unbind(-1)
    b_corners = [(x - 0.5 * w), (y - 0.5 * h),
         (x + 0.5 * w), (y + 0.5 * h)]
    return torch.stack(b_corners, dim=-1).numpy()


def get_iou(bboxes1, bboxes2):
    if len(bboxes1) == 0:
        return np.zeros((bboxes2.shape[0],)), None
    bbox_type = type(bboxes1)
    if bbox_type == ndarray:
        bboxes1 = torch.tensor(bboxes1)
        bboxes2 = torch.tensor(bboxes2)
    area1 = box_area(bboxes1)
    area2 = box_area(bboxes2)

    lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])
    rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:,:,0] * wh[:,:,1]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou.numpy(), union


def generalized_iou(bboxes1, bboxes2):
    bbox_type = type(bboxes1)
    if bbox_type == ndarray:
        bboxes1 = torch.tensor(bboxes1)
        bboxes2 = torch.tensor(bboxes2)

    iou, union = get_iou(bboxes1, bboxes2)

    lt = torch.min(bboxes1[:, None, :2], bboxes2[:, :2])
    rb = torch.max(bboxes1[:, None, 2:], bboxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:,:,0] * wh[:,:,1]

    return iou - (inter - union) / inter


def calc_det_metrics(bboxes, targets, iou_tr=0.3):
    TP = 0
    FP = 0
    FN = len(targets)
    mean_iou = []
    iou, _ = get_iou(targets, bboxes)
    for i in range(len(targets)):
        t_iou = iou[i]
        tr_iou_scores = t_iou[t_iou > iou_tr]
        if len(tr_iou_scores) > 0:
            TP += 1
            FN -= 1
            mean_iou += tr_iou_scores.tolist()
        else:
            FP += 1
            mean_iou.append(0)


    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
#    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    mean_iou = sum(mean_iou) / len(mean_iou) if len(mean_iou) > 0 else 0
    return mean_iou, precision, recall