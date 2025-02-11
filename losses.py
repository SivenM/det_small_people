import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment
from utils import generalized_iou, to_corners
from pprint import pprint


class HungarianMatcher(nn.Module):
    '''
    Реализация венгерского алгоритма для вычисления
    соответствия между предсказаниями и истинными значениями
    '''
    def __init__(self, cost_class:float=1., cost_bbox:float=1., cost_giou:float=1., conf_type='sigmoid', debug=False, device='cuda') -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.debug = debug
        self.conf_type = conf_type
        self.device = device

    @torch.no_grad()
    def forward(self, preds:dict, targets:dict):
        batch_size, num_queries = preds['bbox'].shape[:2]
        if self.conf_type == 'softmax':
            out_conf = preds['logits'].flatten(0, 1).softmax(-1)
        elif self.conf_type == 'sigmoid':
            out_conf = F.sigmoid(preds['logits'].flatten(0, 1))
        out_bbox = preds['bbox'].flatten(0, 1)

        target_labels = torch.cat([v["labels"] for v in targets])
        target_bbox = torch.cat([v["bbox"] for v in targets])
        if self.debug:
            print(f'pred conf: {out_conf}')
            print(f'pred bbox: {out_bbox}')
        #print(f'pred bbox: {out_bbox.shape}\n')
        #print(f'target labels: {target_labels.dtype}')
        #print(f'target bbox: {target_bbox.dtype}')
        #print(f'target bbox: {target_bbox.shape}')
        if self.conf_type == 'softmax':
            cost_class = -out_conf[:, target_labels]
        elif self.conf_type == 'sigmoid':
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1-alpha) * (out_conf ** gamma) * (-(1 - out_conf + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_conf) ** gamma) * (- (out_conf + 1e-8).log())
            cost_class = pos_cost_class[target_labels] - neg_cost_class[target_labels]
        cost_bbox = torch.cdist(out_bbox, target_bbox, p=1)
        cost_giou = -generalized_iou(to_corners(out_bbox), to_corners(target_bbox))
        #print(f'cost iou: {cost_giou}')

        cost = self.cost_bbox*cost_bbox + self.cost_class*cost_class + self.cost_giou*cost_giou
        cost = cost.view(batch_size, num_queries, -1).cpu()
        sizes = [len(t['bbox']) for t in targets]
        if self.debug:
            print(f'cost_class: {cost_class.shape}')
            print(f'cost_bbox: {cost_bbox.shape}')
            print(f'cost: {cost.dtype}')
            print(f'cost: {cost.shape}')
            pprint(sizes)
            indices = []
            for i, c in enumerate(cost.split(sizes, -1)):
                if self.debug:
                    print(c[i])
                    print(c[i].shape)
                    print(c[i].dtype)
                    ind = linear_sum_assignment(c[i])
                    print('====')
                else:
                    ind = linear_sum_assignment(c[i])
                indices.append(ind)
        else:
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(sizes, -1))]
        indices_torch = [(torch.as_tensor(i, dtype=torch.int64, device=self.device), torch.as_tensor(j, dtype=torch.int64, device=self.device)) for i, j in indices]
        return indices_torch
    

class DetrLoss(nn.Module):

    def __init__(self, 
                 num_classes:int, 
                 matcher:HungarianMatcher, 
                 cls_scale:float=0.1, 
                 bbox_scale:float=5, 
                 giou_scale:float=2,
                 conf_type='sigmoid',
                 debug:bool=False) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.cls_scale = cls_scale
        self.bbox_scale = bbox_scale
        self.giou_scale = giou_scale
        ce_weights = torch.ones(2)
        ce_weights[-1] = self.cls_scale
        self.conf_type = conf_type,
        self.register_buffer('ce_weights', ce_weights.to('cuda'))
        self.debug = debug

    def _get_pred_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        pred_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, pred_idx

    def _get_target_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    def loss_cls(self, preds, targets, indices, num_bboxes):
        if self.debug:
            print(f'{self.conf_type=}')
            print(f'{type(self.conf_type)=}')
        idx = self._get_pred_permutation_idx(indices)
        logits = preds['logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        if 'softmax' in self.conf_type:
            target_classes = torch.full(logits.shape[:2], self.num_classes,
                                        dtype=torch.long, device=logits.device)
            target_classes[idx] = target_classes_o
            target_classes = target_classes 
            ce_loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), target_classes, self.ce_weights)
        elif 'sigmoid' in self.conf_type:
            target_classes = torch.full(logits.shape[:2], 0,
                                        dtype=torch.long, device=logits.device)
            target_classes[idx] = target_classes_o
            ce_loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(logits), target_classes.to(torch.float32))
        return ce_loss
    
    def loss_bbox(self, preds, targets, indices, num_bboxes):
        idx = self._get_pred_permutation_idx(indices)
        bboxes = preds['bbox'][idx]
        #print(f'bboxes: {bboxes}\n')
        #print(f'bboxes xyxy: {to_corners(bboxes)}')
        target_bboxes = torch.cat([t['bbox'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        #print(f'pred shape: {bboxes.shape}')
        #print(f'target shape: {target_bboxes.shape}')

        loss_l1 = nn.functional.l1_loss(bboxes, target_bboxes, reduction='none')
        iou = generalized_iou(
            to_corners(bboxes),
            to_corners(target_bboxes),
        )
        loss_giou = 1 - torch.diag(iou)
        loss_l1 = loss_l1.sum() / num_bboxes
        loss_giou = loss_giou.sum() / num_bboxes
        if self.debug:
            print(f'bbox loss: {loss_l1}')
            print(f'bbox loss iou: {loss_giou}')

        out_loss = loss_l1 * self.bbox_scale + loss_giou * self.giou_scale
        return out_loss, iou
    
    def forward(self, preds, targets):
        preds_without_aux = {k: v for k, v in preds.items() if k != 'aux_outputs'}
        indices = self.matcher(preds_without_aux, targets)
        num_bboxes = sum(len(t['labels']) for t in targets)
        c_loss = self.loss_cls(preds, targets, indices, num_bboxes)
        b_loss, iou = self.loss_bbox(preds, targets, indices, num_bboxes)
        if self.debug:
            print(f'cls loss: {c_loss}')
            print(f'bbox loss: {b_loss}')

        losses = c_loss + b_loss
        if 'aux_outputs' in preds:
            for i, aux_outputs in enumerate(preds['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                c_loss = self.loss_cls(preds, targets, indices, num_bboxes)
                b_loss, iou = self.loss_bbox(preds, targets, indices, num_bboxes)
                aux_loss = c_loss + b_loss
                losses = aux_loss
        return losses, iou


class LocLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

    def forward(self, preds:torch.Tensor, targets:list):
        pass
        ##print(f'target_bboxes shape: {target_bboxes.shape}')
        #loss_l1 = torch.nn.functional.l1_loss(pred_bboxes, target_bboxes, reduction='none')
        #return loss_l1.sum() / target_bboxes.shape[0]
    

class DetrLocLoss(nn.Module):
    """
    matcher + L1
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _get_pred_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        pred_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, pred_idx
    
    @torch.no_grad()
    def match(self, preds:torch.Tensor, targets:list):
        bs, num_q = preds.shape[:2]
        pred = preds.flatten(0, 1)
        gt = torch.cat([bbox for bbox in targets])
        cost = torch.cdist(pred, gt, p=1)
        C = cost.view(bs, num_q, -1).cpu()
        sizes = [len(bbox) for bbox in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices_torch = [(torch.as_tensor(i, dtype=torch.int64, device='cuda'), torch.as_tensor(j, dtype=torch.int64, device='cuda')) for i, j in indices]
        return indices_torch
    
    def forward(self, preds:torch.Tensor, targets:list):
        indices = self.match(preds, targets)
        idx = self._get_pred_permutation_idx(indices)
        pred_bboxes = preds[idx]
        target_bboxes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        #print(f'target_bboxes shape: {target_bboxes.shape}')
        loss_l1 = torch.nn.functional.l1_loss(pred_bboxes, target_bboxes, reduction='none')
        return loss_l1.sum() / target_bboxes.shape[0]
    

class DetrLocLossV2(nn.Module):
    """
    matcher + L1 + GIoU
    """
    def __init__(self, l1_w=5., giou_w=2.) -> None:
        super().__init__()
        self.l1_w=l1_w
        self.giou_w=giou_w

    def _get_pred_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        pred_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, pred_idx
    
    @torch.no_grad()
    def match(self, preds:torch.Tensor, targets:list):
        bs, num_q = preds.shape[:2]
        pred = preds.flatten(0, 1)
        gt = torch.cat([bbox for bbox in targets])
        l1_cost = torch.cdist(pred, gt, p=1)
        giou_cost = -generalized_iou(to_corners(pred), to_corners(gt))
        cost = l1_cost * self.l1_w + giou_cost * self.giou_w
        C = cost.view(bs, num_q, -1).cpu()
        sizes = [len(bbox) for bbox in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices_torch = [(torch.as_tensor(i, dtype=torch.int64, device='cuda'), torch.as_tensor(j, dtype=torch.int64, device='cuda')) for i, j in indices]
        return indices_torch
    
    def forward(self, preds:torch.Tensor, targets:list):
        num_bboxes = sum(len(bboxes) for bboxes in targets)
        indices = self.match(preds, targets)
        idx = self._get_pred_permutation_idx(indices)
        pred_bboxes = preds[idx]
        target_bboxes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        #print(f'target_bboxes shape: {target_bboxes.shape}')
        loss_l1 = torch.nn.functional.l1_loss(pred_bboxes, target_bboxes, reduction='none')
        loss_giou = 1 - torch.diag(generalized_iou(
            to_corners(pred_bboxes),
            to_corners(target_bboxes),
        ))
        l1 = loss_l1.sum() / num_bboxes
        giou = loss_giou.sum() / num_bboxes
        out = l1 * self.l1_w + giou * self.giou_w
        return out