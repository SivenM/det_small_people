import os
import yaml
from data_perp import DETRDataset, DetrLocDataset, SeqCls, LocDataLoader
from models import DETR, DetrLoc, MyVanilaDetr, SeqDetrLoc
from coaching import Coach, DetrLocCoach, LocCoach
import losses
import utils

import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import argparse


def collate_fn(batch):
    samples = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return samples, targets


def create_dataloader(path:str, mean:list, std:list, batch_size=8, model_type:str='detr', norm:bool=True, rgb:bool=False):
    sample_transforms = v2.Compose([
        #transforms.ToTensor(),
        v2.Lambda(lambda x: torch.tensor(x)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean,
        std=std )
        ])
    if model_type == 'detr':
        dataset = DETRDataset(
            path, 
            norm=False, 
            transform=sample_transforms
            )
    elif model_type == 'detr_loc':
        dataset = DetrLocDataset(
            path, 
            norm=norm, 
            transform=sample_transforms
            )
    elif model_type == 'detr_loc_1_frame':
        sample_transforms = v2.Compose([
        v2.ToTensor(),
        #v2.Lambda(lambda x: torch.tensor(x)),
        v2.Normalize(mean=mean,
        std=std )
        ])
        dataset = DetrLocDataset(
            path, 
            norm=norm, 
            transform=sample_transforms
            )
    elif model_type == 'vanilla_detr':
        print('+')
        sample_transforms = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
        ])
        dataset = DetrLocDataset(
            path, 
            norm=norm, 
            transform=sample_transforms,
            rgb=rgb
            )
    elif model_type == 'seq_detr_loc':
        sample_transforms = v2.Compose([
        v2.Lambda(lambda x: torch.tensor(x, dtype=torch.float32) / 255.)
        #v2.ToImage(),
        #v2.ToDtype(torch.float32, scale=True),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406,0.485, 0.456, 0.406, 0.406],
        #std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.225] )
        ])
        target_transforms = v2.Lambda(lambda x: torch.tensor([x], dtype=torch.float32)) 
        dataset = DetrLocDataset(
            path, 
            norm=norm, 
            transform=sample_transforms
            )
    elif model_type == 'classic_loc':
        sample_transforms = v2.Compose([
        v2.Lambda(lambda x: torch.tensor(x, dtype=torch.float32) / 255.)
        ])
        dataset = LocDataLoader(
            path, 
            norm=norm, 
            transform=sample_transforms,
            max_bboxes=7
            )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    return dataloader


def print_params(cfg:dict, num_train_data:int, num_val_data:int):
    print(f'\n\nparams:\n')
    for k,v in cfg.items():
        print(f'{k}: {v}')

    print(f'num_train_data: {num_train_data}')
    print(f'num_val_data: {num_val_data}')
    print('\n')


def train(cfg:dict):
    if cfg['model_type'] == 'detr':
        coach = Coach(cfg['name'], cfg['save_dir'], tboard=cfg['tb'], debug=cfg['debug'])
        train_model = DETR(
            cfg['num_encoder_blocks'],
            cfg['num_decoder_blocks'],
            cfg['num_queries'],
            cfg['num_cls'],
            cfg['emb_dim'],
            cfg['img_size'],
            cfg['num_imgs'],
            cfg['patch_size']
        )
        matcher = losses.HungarianMatcher(cfg['cost_class'], cfg['cost_bbox'], cfg['cost_giou'])
        loss_fn = losses.DetrLoss(cfg['num_cls'], matcher, cfg['cls_scale'], cfg['bbox_scale'], cfg['giou_scale'])
    elif cfg['model_type'] in ['detr_loc', 'detr_loc_1_frame']:
        coach = DetrLocCoach(cfg['name'], cfg['save_dir'], tboard=cfg['tb'], debug=cfg['debug'])
        train_model = DetrLoc(
            cfg['num_encoder_blocks'],
            cfg['num_decoder_blocks'],
            cfg['num_queries'],
            cfg['num_cls'],
            cfg['emb_dim'],
            cfg['img_size'],
            cfg['num_imgs'],
            cfg['patch_size']
        )
        loss_fn = losses.DetrLocLoss()

    elif cfg['model_type'] == 'vanilla_detr':
        coach = DetrLocCoach(cfg['name'], cfg['save_dir'], tboard=cfg['tb'], debug=cfg['debug'])
        train_model = MyVanilaDetr(
            cfg['emb_dim'],
            8,
            cfg['num_encoder_blocks'],
            cfg['num_decoder_blocks'],
        )
        loss_fn = losses.DetrLocLossV2()
    elif cfg['model_type'] == 'seq_detr_loc':
        coach = DetrLocCoach(cfg['name'], cfg['save_dir'], tboard=cfg['tb'], debug=cfg['debug'])
        train_model = SeqDetrLoc(
            cfg['emb_dim'],
            cfg['num_queries'],
            cfg['pos_per_block'],
            cfg['num_encoder_blocks'],
            cfg['num_decoder_blocks'],
            cfg['num_conv_layers'],
            cfg['out_channel_outputs'],
            cfg['patch_size'],
            cfg['num_imgs'],
            cfg['max_seq']
        )
        loss_fn = losses.DetrLocLossV2()

    elif cfg['model_type'] == 'classic_loc':
        coach = LocCoach(cfg['name'], cfg['save_dir'], tboard=cfg['tb'], debug=cfg['debug'])
        train_model = SeqDetrLoc(
            cfg['emb_dim'],
            cfg['num_queries'],
            cfg['pos_per_block'],
            cfg['num_encoder_blocks'],
            cfg['num_decoder_blocks'],
            cfg['num_conv_layers'],
            cfg['out_channel_outputs'],
            cfg['patch_size'],
            cfg['num_imgs'],
            cfg['max_seq']
        )
        loss_fn = torch.nn.L1Loss()
        
    else:
        raise "Wrong model type. Only detr or detr_loc"
    
    train_dataloader = create_dataloader(cfg['train_dataset_path'], cfg['mean'], cfg['std'], cfg['train_batch_size'], cfg['model_type'], cfg['norm'], cfg['rgb'])
    val_dataloader = create_dataloader(cfg['val_dataset_path'], cfg['mean'], cfg['std'], cfg['val_batch_size'], cfg['model_type'], cfg['norm'], cfg['rgb'])

    print_params(cfg, len(train_dataloader), len(val_dataloader))
    coach.fit(
        cfg['epoches'],
        train_model,
        loss_fn,
        cfg['lr'],
        train_dataloader,
        val_dataloader
        )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='скрипт для обучения DETR')
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
