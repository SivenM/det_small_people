import sys
sys.path.append('.')

import torch
from torch import nn
from models.detr_zoo import DefEncoderDet, DDBackbone



num_queries = 5
num_feature_levels = 4
bs = 16
backbone = DDBackbone()
model = DefEncoderDet(backbone, 5, 4, 256, 12)
x = torch.randn((bs,1,224,224))
pred = model(x)

print(model)