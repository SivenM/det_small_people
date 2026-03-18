import sys
from data_perp import CropDatasetV2
from coaching import Coach
from models.transformer import VisTransformer, VisTransformerMean
import argparse
import yaml
import torch
from torchvision import transforms
from torch.utils.data import DataLoader


def print_cfg(config:dict):
    print('\nconfig:')
    for k, v in config.items():
        print(f'{k}: {v}')
    print('\n')


def get_args():
    parser = argparse.ArgumentParser(prog="Скрипт обучения для моделей классификации последовательностей")
    parser.add_argument("-c", "--config", type=argparse.FileType('r'), default=None, help="Путь до конфига")
    args = parser.parse_args()
    if args.config:
        config = yaml.safe_load(args.config)
        return config
    else:
        return


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
    config_list = get_args()
    if config_list:
        main(config_list)
    else:
        print("\nConfig not not found. Use `python train_obj.py -c config.yaml`")
    