import os
import yaml
from data_perp import SeqCls
from models.transformer import SeqModel, CCTransformer, CCTransformerV2
from coaching import Coach, DetrLocCoach

import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import argparse


def print_params(cfg:dict, num_train_data:int, num_val_data:int):
    print(f'\n\nparams:\n')
    for k,v in cfg.items():
        print(f'{k}: {v}')

    print(f'num_train_data: {num_train_data}')
    print(f'num_val_data: {num_val_data}')
    print('\n')


def create_dataloader(path:str, batch_size=8) -> DataLoader:
    sample_transforms = v2.Compose([
        v2.Lambda(lambda x: torch.tensor(x, dtype=torch.float32) / 255.)
        #v2.ToImage(),
        #v2.ToDtype(torch.float32, scale=True),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406,0.485, 0.456, 0.406, 0.406],
        #std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.225] )
        ])
    target_transforms = v2.Lambda(lambda x: torch.tensor([x], dtype=torch.float32)) 
    dataset = SeqCls(
        path,
        sample_transforms,
        target_transforms,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    

def train(cfg:dict) -> None:
    train_dataloader = create_dataloader(cfg['train_dataset'], cfg['train_batch_size'])
    val_dataloader = create_dataloader(cfg['val_dataset'], cfg['val_batch_size'])

    coach = Coach(cfg['name'], cfg['save_dir'], device='cuda')
    #model = SeqModel(1, cfg['num_blocks'])
    model = CCTransformerV2(
        num_classes=1, 
        emb_dim=cfg['emb_dim'], 
        num_blocks=cfg['num_blocks'], 
        num_conv_layers=cfg['num_conv_layers'], 
        out_channel_outputs=cfg['out_ch_outputs'], 
        patch_size=cfg['patch_size'], C=cfg['frame_rate'], 
        max_seq=cfg['max_seq'])
    loss_fn = torch.nn.BCELoss()

    print_params(cfg, len(train_dataloader), len(val_dataloader))
    coach.fit(
        cfg['epoches'],
        model,
        loss_fn,
        cfg['lr'],
        train_dataloader,
        val_dataloader
        )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='скрипт для обчения классификации последовательностей')
    parser.add_argument('-c', '--config', dest='config', type=argparse.FileType('r'), default=None)
    args = parser.parse_args()
    if args.config:
        try:
            config = yaml.safe_load(args.config)
            train(config)
        except FileNotFoundError:
            print(f'Файл {args.config} не найден')
        except yaml.YAMLError as exc:
            print(f'Ошибка при чтении YAML файла: {exc}')