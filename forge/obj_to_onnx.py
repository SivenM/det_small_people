import torch
from torch import nn
import sys
sys.path.append('.')
from models.transformer import VisTransformer, VisTransformerMean


def load_model(weights_path:str, model_type:str) -> nn.Module:
    if model_type == 'mean':
        model = VisTransformerMean(config['num_blocks'], num_cls=config['num_cls'], input_c=config['input_c'])
    elif model_type == 'seqpool':
        model = VisTransformer(config['num_blocks'], num_cls=config['num_cls'], input_c=config['input_c'])
    else:
        raise f"wrong model type {model_type}. Should be 'mean' or 'seqpool'"
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model

def main(config:dict):
    dummy_input = torch.randn(1, config['input_c'], config['input_h'], config['input_w'])
    model = load_model(config['weights'], config['model_type'])
    out = model(dummy_input)
    #print(model.head[0].weight.shape)
    print(out)
    print(out.shape)
    torch.onnx.export(
        model,
        dummy_input,
        config['save_path'],
        input_names=['inputs'],
    #    output_names=['outputs'],
    #    dynamic_axes={
    #        "inputs": {0: "bath_size"},
    #        "outputs": {0: "bath_size"}
    #    },
        opset_version=16, 
        verbose=False,
    )
    print('done!')


if __name__ == '__main__':
    config= {
        'model_type': 'seqpool', # mean or seqpool
        'weights': "/home/max/ieos/small_obj/vid_pred/runs/sequense_cls/exp_no_imagenet_norm/obj_VitSeqPool_10_1_1blocks/models/last_acc0.9477040816326531.pth",
        'num_blocks': 1,
        'num_cls': 4,
        'input_c': 10,
        'input_w': 50,
        'input_h': 50,
        'save_path': '/home/max/ieos/small_obj/vid_pred/onnx_models/obj_VitSeqPool_10_1_1blocks.onnx'
    }
    main(config)