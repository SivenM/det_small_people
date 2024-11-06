import torch
from torch import nn
from fvcore.nn import FlopCountAnalysis
from models import transformer

def load_weights(model:nn.Module, weight_path:str):
    model.load_state_dict(torch.load(weight_path))
    model.to('cuda')
    model.eval()
    return model

def main(weight_path:str, num_blocks:int=1):
    model = transformer.VisTransformer(num_blocks)
    model = load_weights(model, weight_path)
    input_x = torch.randn(1,10,50,50).to('cuda')
    flops = FlopCountAnalysis(model, input_x)
    #y = model(input_x)
    #print(y.shape)
    print(f"FLOPs: {flops.total()}")


if __name__ == "__main__":
    weight_path = "/home/max/ieos/small_obj/vid_pred/runs/exp2.1_obj_nblocks1_ep50/models/best_loss_0.06.pth"
    num_blocks = 1
    main(weight_path, num_blocks)  