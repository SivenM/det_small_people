import os
import yaml
from data_perp import DETRDataset, DetrLocDataset, SeqCls, LocDataLoader
from models.detr_zoo import DETR, DetrLoc, MyVanilaDetr, SeqDetrLoc, CCTransformerLoc, CCTransformerDet, TimeSformerDet, DeformableDETR, DefEncoderDet
from models.transformer import DeformableTransformer
from models import backbones
from coaching import Coach, DetrLocCoach, LocCoach, SeqDetCoach
import losses
import utils

import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import argparse


def load_weights(path:str, model:torch.nn.Module) -> torch.nn.Module:
    return model.load_state_dict(torch.load(path), strict=False)


def collate_fn(batch):
    samples = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return samples, targets


def create_dataloader(path:str, mean:list, std:list, batch_size=8, model_type:str='detr', norm:bool=True, r_size:list=[], mode:bool='base'):
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
            rgb=True
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
    elif model_type in ['classic_loc', 'classic_seq_loc']:
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
    elif model_type == 'classic_seq_det':
        sample_transforms = v2.Compose([
            v2.Lambda(lambda x: torch.tensor(x, dtype=torch.float32) / 255.)
        ])
        dataset = DETRDataset(
            path, 
            norm=norm, 
            transform=sample_transforms
            )
    elif model_type == 'seq_det_one_frame':
        sample_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        dataset = DETRDataset(
            path, 
            norm=True, 
            transform=sample_transforms
            )
    elif model_type == 'timesformer':
        sample_transforms = v2.Compose([
            v2.Lambda(lambda x: torch.tensor(x, dtype=torch.float32) / 255.)
        ])
        dataset = DETRDataset(
            path, 
            norm=norm, 
            transform=sample_transforms,
            mode='time',
            )
    elif model_type == 'def_detr' or model_type == 'def_endet':
        sample_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            #v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dataset = DETRDataset(
            path, 
            norm=True, 
            transform=sample_transforms,
            size=r_size,
            mode=mode
            )
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
    
    elif cfg['model_type'] == 'classic_seq_loc':
        coach = LocCoach(cfg['name'], cfg['save_dir'], tboard=cfg['tb'], debug=cfg['debug'])
        train_model = CCTransformerLoc(
        num_bboxes=cfg['max_bboxes'], 
        emb_dim=cfg['emb_dim'], 
        num_blocks=cfg['num_encoder_blocks'], 
        num_conv_layers=cfg['num_conv_layers'], 
        out_channel_outputs=cfg['out_channel_outputs'], 
        patch_size=cfg['patch_size'], C=cfg['num_imgs'], 
        max_seq=cfg['max_seq']
        )
        loss_fn = torch.nn.L1Loss()
    elif cfg['model_type'] in ['classic_seq_det', 'seq_det_one_frame']:
        coach = SeqDetCoach(cfg['name'], cfg['save_dir'], tboard=cfg['tb'], debug=cfg['debug'], progress_bar=cfg['progress_bar'])
        train_model = CCTransformerDet(
            num_bboxes=cfg['max_bboxes'], 
            emb_dim=cfg['emb_dim'], 
            num_blocks=cfg['num_encoder_blocks'], 
            num_conv_layers=cfg['num_conv_layers'], 
            out_channel_outputs=cfg['out_channel_outputs'], 
            patch_size=cfg['patch_size'], C=cfg['num_imgs'], 
            max_seq=cfg['max_seq']
        )
        matcher = losses.HungarianMatcher(cfg['cost_class'], cfg['cost_bbox'], cfg['cost_giou'])
        loss_fn = losses.DetrLoss(cfg['num_cls'], matcher, cfg['cls_scale'], cfg['bbox_scale'], cfg['giou_scale'])
    elif cfg['model_type'] == 'timesformer':
        coach = SeqDetCoach(cfg['name'], cfg['save_dir'], tboard=cfg['tb'], debug=cfg['debug'], progress_bar=cfg['progress_bar'])
        train_model = TimeSformerDet(
            num_bboxes=cfg['max_bboxes'], 
            emb_dim=cfg['emb_dim'], 
            num_blocks=cfg['num_encoder_blocks'], 
            num_output_channels=cfg['out_channel_outputs'], 
            num_patches=cfg['num_patches'],
            num_frames=cfg['num_imgs'], 
            num_heads=4,
        )
        matcher = losses.HungarianMatcher(cfg['cost_class'], cfg['cost_bbox'], cfg['cost_giou'])
        loss_fn = losses.DetrLoss(cfg['num_cls'], matcher, cfg['cls_scale'], cfg['bbox_scale'], cfg['giou_scale'])
    elif cfg['model_type'] == 'def_detr':
        coach = SeqDetCoach(cfg['name'], cfg['save_dir'], tboard=cfg['tb'], debug=cfg['debug'], progress_bar=cfg['progress_bar'])
        if cfg['backbone'] == 'convnext':
            backbone = backbones.ConvNextBackbone()
        else:
            backbone = backbones.DDBackbone()
        transformer = DeformableTransformer(emb_dim=cfg['emb_dim'], nhead=4, 
                                            num_encoder_layers=cfg['num_encoder_blocks'], num_decoder_layers=cfg['num_decoder_blocks'],
                                            return_intermediate_dec=True)
        train_model = DeformableDETR(backbone, transformer, cfg['num_cls'], cfg['num_queries'], cfg['num_feature_levels'], cfg['aux_loss'])
        matcher = losses.HungarianMatcher(cfg['cost_class'], cfg['cost_bbox'], cfg['cost_giou'])
        loss_fn = losses.DetrLoss(cfg['num_cls'], matcher, cfg['cls_scale'], cfg['bbox_scale'], cfg['giou_scale'])
    elif cfg['model_type'] == 'def_endet':
        coach = SeqDetCoach(cfg['name'], cfg['save_dir'], tboard=cfg['tb'], debug=cfg['debug'], progress_bar=cfg['progress_bar'])
        if cfg['backbone'] == 'convnext':
            backbone = backbones.ConvNextBackbone()
        else:
            backbone = backbones.DDBackbone()
        train_model = DefEncoderDet(backbone, cfg['num_queries'], cfg['num_feature_levels'], cfg['emb_dim'], cfg['num_encoder_blocks'])
        matcher = losses.HungarianMatcher(cfg['cost_class'], cfg['cost_bbox'], cfg['cost_giou'])
        loss_fn = losses.DetrLoss(cfg['num_cls'], matcher, cfg['cls_scale'], cfg['bbox_scale'], cfg['giou_scale'])
    else:
        raise "Wrong model type."
    
    train_dataloader = create_dataloader(cfg['train_dataset_path'], cfg['mean'], 
                                        cfg['std'], cfg['train_batch_size'],
                                        cfg['model_type'], cfg['norm'], 
                                        cfg['r_size'], cfg['mode'])
    val_dataloader = create_dataloader(cfg['val_dataset_path'], cfg['mean'],
                                    cfg['std'], cfg['val_batch_size'], 
                                    cfg['model_type'], cfg['norm'], 
                                    cfg['r_size'], cfg['mode'])

    if len(cfg['from_save']):
        train_model = load_weights(cfg['from_save'], train_model)

    print_params(cfg, len(train_dataloader), len(val_dataloader))
    coach.fit(
        cfg['epoches'],
        train_model,
        loss_fn,
        cfg['lr'],
        train_dataloader,
        val_dataloader,
        cfg['start_epoch']
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
