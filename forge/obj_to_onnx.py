import torch
from torch import nn
import sys
sys.path.append('.')
from models.transformer import VisTransformer, VisTransformerMean, TimeSfromerCls
import argparse
import yaml
from pprint import pprint


def get_args():
    parser = argparse.ArgumentParser(prog='Конвортирует модели в onnx формат')
    parser.add_argument('-c', '--config', default=None, type=argparse.FileType('r'), help='путь до конфига')
    args = parser.parse_args()
    if args.config:
        try:
            config = yaml.safe_load(args.config)
            return config
        except FileNotFoundError:
            print(f'Файл {args.config} не найден')
        except yaml.YAMLError as exc:
            print(f'Ошибка при чтении YAML файла: {exc}')
    else:
        return
    

def load_model(config:dict) -> nn.Module:
    if config['model_type'] == 'mean':
        model = VisTransformerMean(config['num_blocks'], num_cls=config['num_cls'], input_c=config['input_c'])
    elif config['model_type'] == 'seqpool':
        model = VisTransformer(config['num_blocks'], num_cls=config['num_cls'], input_c=config['input_c'])
    elif config['model_type'] == 'timesformer':
        model = TimeSfromerCls(
            num_blocks=config['num_blocks'],
            num_cls=config['num_cls'],
            patch_size=config['patch_size'],
            num_frames=config['len_seq'],
            num_patches=(config['pad_size'][0] * config['pad_size'][1])//config['patch_size']**2,
            num_heads=config['num_heads'],
            hidden_dim=config['hidden_dim']
        )
    else:
        raise f"wrong model type {config['model_type']}. Should be 'mean', 'seqpool' or 'timesformer'"
    model.load_state_dict(torch.load(config['model_path']))
    model.eval()
    return model


def create_input(size:list):
    return torch.randn(*size)


def main(config:dict):
    print('config params:')
    pprint(config)
    print('-'*80,'\n')
    print('start converting...')
    dummy_input = create_input(config['input_size'])
    model = load_model(config)
    out = model(dummy_input)
    #print(model.head[0].weight.shape)
    #print(out)
    print(f'model out shape: {out.shape}')
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
    #config= {
    #    'model_type': 'seqpool', # mean or seqpool
    #    'weights': "/home/max/ieos/small_obj/vid_pred/runs/sequense_cls/exp_no_imagenet_norm/obj_VitSeqPool_10_1_1blocks/models/last_acc0.9477040816326531.pth",
    #    'num_blocks': 1,
    #    'num_cls': 4,
    #    'input_c': 10,
    #    'input_w': 50,
    #    'input_h': 50,
    #    'save_path': '/home/max/ieos/small_obj/vid_pred/onnx_models/obj_VitSeqPool_10_1_1blocks.onnx'
    #}
    config = get_args()
    if config:
        main(config)