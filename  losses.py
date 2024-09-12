import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from utils import generalized_iou, to_corners


class HungarianMatcher(nn.Module):
    '''
    Реализация венгерского алгоритма для вычисления
    соответствия между предсказаниями и истинными значениями
    '''
    def __init__(self, cost_class:float=1., cost_bbox:float=1., cost_giou:float=1.) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, preds:dict, targets:dict):
        batch_size, num_queries = preds['logits'].shape[:2]

        out_conf = preds['logits'].flatten(0, 1).softmax(-1)
        out_bbox = preds['bbox'].flatten(0, 1)
        target_labels = targets['labels']
        target_bbox = targets['bbox']

        cost_class = -out_conf[:, target_labels]
        cost_bbox = torch.cdist(out_bbox, target_bbox, p=1)
        cost_giou = -generalized_iou(to_corners(out_bbox), to_corners(target_bbox))

        cost = self.cost_bbox*cost_bbox + self.cost_class*cost_class + self.cost_giou*cost_giou
        cost = cost.view(batch_size, num_queries, -1)

        sizes = [len(bbox) for bbox in preds['bbox']]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(sizes, -1))]
        indices_torch = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        return indices_torch
    

class DetrLoss(nn.Module):

    def __init__(self, 
                 num_classes:int, 
                 matcher:HungarianMatcher, 
                 cls_scale:float=0.1, 
                 bbox_scale:float=5, 
                 giou_scale:float=2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.cls_scale = cls_scale
        self.bbox_scale = bbox_scale
        self.giou_scale = giou_scale
        ce_weights = torch.ones(num_classes + 1)
        ce_weights[-1] = self.cls_scale
        self.register_buffer('ce_weights', ce_weights)

    def _get_pred_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        pred_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, pred_idx

    def _get_target_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    def loss_cls(self, preds, targets, indices, num_bboxes):
        idx = self._get_pred_permutation_idx(indices)
        logits = preds['logits'][idx]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=logits.device)
        target_classes[idx] = target_classes_o
        ce_loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), target_classes, self.ce_weight)
        return ce_loss
    
    def loss_bbox(self, preds, targets, indices, num_bboxes):
        idx = self._get_pred_permutation_idx(indices)
        bboxes = preds['bbox'][idx]
        target_bboxes = torch.cat([t['bbox'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_l1 = nn.functional.l1_loss(bboxes, target_bboxes, reduction='none')
        loss_giou = 1 - torch.diag(generalized_iou(
            to_corners(bboxes),
            to_corners(target_bboxes),
        ))
        loss_l1 = loss_l1.sum() / num_bboxes
        loss_giou = loss_giou.sum() / num_bboxes
        return loss_l1 * self.bbox_scale + loss_giou * self.giou_scale
    
    def forward(self, preds, targets):
        indices = self.matcher(preds, targets)
        num_bboxes = sum(len(t) for t in targets['labels'])
        c_loss = self.loss_cls(preds, targets, indices, num_bboxes)
        b_loss = self.loss_bbox(preds, targets, indices, num_bboxes)

        losses = c_loss + b_loss
        return losses



