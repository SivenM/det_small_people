import torch
from torch import nn
from fvcore.nn import FlopCountAnalysis
from models import detr_zoo

def load_weights(model:nn.Module, weight_path:str):
    model.load_state_dict(torch.load(weight_path))
    model.to('cuda')
    model.eval()
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(weight_path:str, num_blocks:int=1):
    #model = transformer.VisTransformer(num_blocks)
    #model = load_weights(model, weight_path)
    model = detr_zoo.TimeSformerDet(
            num_bboxes=7, 
            emb_dim=256, 
            num_blocks=5, 
            patch_size=16, 
            num_patches=1200,
            num_frames=10, 
            num_heads=4,
        ).to('cuda')
    input_x = torch.randn(1,10,1,480,640).to('cuda')
    flops = FlopCountAnalysis(model, input_x)
    num_params = count_parameters(model)
    #y = model(input_x)
    #print(y.shape)
    print(f"FLOPs: {flops.total()}")
    print(f'num parameters: {num_params}')
if __name__ == "__main__":
    weight_path = ""#"/home/max/ieos/small_obj/vid_pred/runs/exp2.1_obj_nblocks1_ep50/models/best_loss_0.06.pth"
    num_blocks = 1
    main(weight_path, num_blocks)  