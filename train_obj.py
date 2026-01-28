import sys
from data_perp import CropDatasetV2
from coaching import Coach
from models.transformer import VisTransformer, VisTransformerMean
import argparse

import torch
from torchvision import transforms
from torch.utils.data import DataLoader


def print_cfg(config:dict):
    print('\nconfig:')
    for k, v in config.items():
        print(f'{k}: {v}')
    print('\n')


def main(config_list:list[dict]):
    for config in config_list:
        print_cfg(config)
        coach = Coach(config['name'], config['save_dir'], metric='multi_acc')
        if config['model_type'] == 'seqpool':
            model = VisTransformer(
                num_blocks=config['num_blocks'], 
                num_cls=len(config['labels']),
                input_c=config['len_seq']
                )
        elif config['model_type'] == 'mean':
            model = VisTransformerMean(
                num_blocks=config['num_blocks'], 
                num_cls=len(config['labels']),
                input_c=config['len_seq']
                )
        else:
            raise f'wrong model type: {config['model_type']}'
        
        #sample_transforms = transforms.Compose([
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean=[0.485 for _ in range(config['len_seq'])], #[0.485, 0.456, 0.406, 0.485, 0.456, 0.406,0.485, 0.456, 0.406, 0.406],
        #    std=[0.224 for _ in range(config['len_seq'])])          #[0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.225] )
        #    ])
        #target_transforms = transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long)) 
        train_dataset = CropDatasetV2(config['train_data_path'], config['labels'], None, None, len_seq=config['len_seq'])
        val_dataset = CropDatasetV2(config['val_data_path'], config['labels'], None, None, len_seq=config['len_seq'])
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)
        loss_fn = torch.nn.CrossEntropyLoss()
        #optimizator = torch.optim.Adam(model.parameters(), lr=config['lr'])
        try:
            coach.fit(
                config['epoches'],
                model,
                loss_fn,
                config['lr'],
                train_dataloader,
                val_dataloader,

                )
            print('\n')
        except KeyboardInterrupt:
            print(f'train stoped')
            sys.exit()
        
if __name__ == '__main__':
    config_list = [
        {
            'name': 'obj_VitSeqPool_10_1_1blocks',
            'model_type': 'seqpool',
            'train_data_path': '/home/max/ieos/data/datasets/window_vid_5_v2.1/gen_10_1_5_v3_clear_train_val/train',
            'val_data_path': '/home/max/ieos/data/datasets/window_vid_5_v2.1/gen_10_1_5_v3_clear_train_val/val',
            'labels': ['false_det', 'human', 'human_dynamic', 'human_static'],
            'save_dir': "/home/max/ieos/small_obj/vid_pred/runs/sequense_cls/exp_no_imagenet_norm",
            'num_blocks': 1,
            'epoches': 100,
            'lr': 0.001,
            'len_seq': 10
        },
        #{
        #    'name': 'obj_VitSeqMean_10_1_4blocks',
        #    'model_type': 'mean',
        #    'train_data_path': '/home/max/ieos/data/datasets/window_vid_5_v2.1/gen_10_1_5_v3_clear_train_val/train',
        #    'val_data_path': '/home/max/ieos/data/datasets/window_vid_5_v2.1/gen_10_1_5_v3_clear_train_val/val',
        #    'labels': ['false_det', 'human', 'human_dynamic', 'human_static'],
        #    'save_dir': "/home/max/ieos/small_obj/vid_pred/runs/sequense_cls",
        #    'num_blocks': 4,
        #    'epoches': 150,
        #    'lr': 0.001,
        #    'len_seq': 10
        #},
        #{
        #    'name': 'exp2_obj_20_1',
        #    'train_data_path': '/home/max/ieos/data/datasets/window_vid_5_v2.1/gen_20_1_5_v3_clear_train_val/train',
        #    'val_data_path': '/home/max/ieos/data/datasets/window_vid_5_v2.1/gen_20_1_5_v3_clear_train_val/val',
        #    'labels': ['false_det', 'human', 'human_dynamic', 'human_static'],
        #    'save_dir': "/home/max/ieos/small_obj/vid_pred/runs/sequense_cls",
        #    'num_blocks': 1,
        #    'epoches': 100,
        #    'lr': 0.001,
        #    'len_seq': 20
        #},
        #{
        #    'name': 'exp2_obj_20_2',
        #    'train_data_path': '/home/max/ieos/data/datasets/window_vid_5_v2.1/gen_20_2_10_v3_clear_train_val/train',
        #    'val_data_path': '/home/max/ieos/data/datasets/window_vid_5_v2.1/gen_20_2_10_v3_clear_train_val/val',
        #    'labels': ['false_det', 'human', 'human_dynamic', 'human_static'],
        #    'save_dir': "/home/max/ieos/small_obj/vid_pred/runs/sequense_cls",
        #    'num_blocks': 1,
        #    'epoches': 100,
        #    'lr': 0.001,
        #    'len_seq': 20
        #},
        #{
        #    'name': 'exp2_obj_5_1',
        #    'train_data_path': '/home/max/ieos/data/datasets/window_vid_5_v2.1/gen_5_1_5_v3_clear_train_val/train',
        #    'val_data_path': '/home/max/ieos/data/datasets/window_vid_5_v2.1/gen_5_1_5_v3_clear_train_val/val',
        #    'labels': ['false_det', 'human', 'human_dynamic', 'human_static'],
        #    'save_dir': "/home/max/ieos/small_obj/vid_pred/runs/sequense_cls",
        #    'num_blocks': 1,
        #    'epoches': 100,
        #    'lr': 0.001,
        #    'len_seq': 5
        #},
        #{
        #    'name': 'exp2_obj_8_1',
        #    'train_data_path': '/home/max/ieos/data/datasets/window_vid_5_v2.1/gen_8_1_5_v3_clear_train_val/train',
        #    'val_data_path': '/home/max/ieos/data/datasets/window_vid_5_v2.1/gen_8_1_5_v3_clear_train_val/val',
        #    'labels': ['false_det', 'human', 'human_dynamic', 'human_static'],
        #    'save_dir': "/home/max/ieos/small_obj/vid_pred/runs/sequense_cls",
        #    'num_blocks': 1,
        #    'epoches': 100,
        #    'lr': 0.001,
        #    'len_seq': 8
        #},
        #{
        #    'name': 'exp2_obj_9_3',
        #    'train_data_path': '/home/max/ieos/data/datasets/window_vid_5_v2.1/gen_9_3_10_v3_clear_train_val/train',
        #    'val_data_path': '/home/max/ieos/data/datasets/window_vid_5_v2.1/gen_9_3_10_v3_clear_train_val/val',
        #    'labels': ['false_det', 'human', 'human_dynamic', 'human_static'],
        #    'save_dir': "/home/max/ieos/small_obj/vid_pred/runs/sequense_cls",
        #    'num_blocks': 1,
        #    'epoches': 100,
        #    'lr': 0.001,
        #    'len_seq': 9
        #},
        
    ]

    main(config_list)