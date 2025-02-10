import sys
sys.path.append('.')

import torch
from torch import nn
import timm
from pprint import pprint

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = timm.create_model('convnextv2_nano', features_only=True)
print(model, '\n')
print(f'num model params{count_parameters(model)}')
x = torch.randn((2,3,224,224))
y = model(x)
print(len(y))
for el in y:
    print(f'\ty el: {el.shape}')
pprint(timm.list_models())
#print(f'model out: {y.shape}')
#print(model.stages_3)
#print(dir(model.stages_1))