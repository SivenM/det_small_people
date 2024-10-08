import os
import yaml
from data_perp import DETRDataset
from models import DETR
from coaching import Coach
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


def create_dataloader(path:str, mean:list, std:list, batch_size=8):
    sample_transforms = v2.Compose([
        #transforms.ToTensor(),
        v2.Lambda(lambda x: torch.tensor(x)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean,
        std=std )
        ])
    
    dataset = DETRDataset(
        path, 
        norm=False, 
        transform=sample_transforms
        )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    return dataloader


def train(cfg:dict):
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
    train_dataloader = create_dataloader(cfg['train_dataset_path'], cfg['mean'], cfg['std'], cfg['train_batch_size'])
    val_dataloader = create_dataloader(cfg['val_dataset_path'], cfg['mean'], cfg['std'], cfg['val_batch_size'])
    matcher = losses.HungarianMatcher(cfg['cost_class'], cfg['cost_bbox'], cfg['cost_giou'])
    loss_fn = losses.DetrLoss(cfg['num_cls'], matcher, cfg['cls_scale'], cfg['bbox_scale'], cfg['giou_scale'])
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
