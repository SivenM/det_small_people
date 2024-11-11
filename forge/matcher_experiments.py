import torch


def calc_sigmoid_cost():
    out_prob = torch.randn((16, 10)).flatten(0, 1).sigmoid()
    tgt_labels = torch.ones((100), dtype=torch.long)
    print(f'out prob: {out_prob.shape}\ntgt_labels: {tgt_labels.shape}\n')

    alpha = 0.25
    gamma = 2.0
    neg_cost_class = (1-alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
    pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (- (out_prob + 1e-8).log())
    print(f'neg_cost_class: {neg_cost_class.shape}\npos_cost_class: {pos_cost_class.shape}')
    cost_class = pos_cost_class[tgt_labels] - neg_cost_class[tgt_labels]
    print(f'cost_class: {cost_class.shape}')

    out_bbox = torch.randn(160, 4)
    tgt_bboxes = torch.randn(100, 4)
    box_cost = torch.cdist(out_bbox, tgt_bboxes, p=1)
    print(f'\nbox cost: {box_cost.shape}')

    C = cost_class + box_cost
    print(f'\n final cost: {C.shape}')

    
calc_sigmoid_cost()