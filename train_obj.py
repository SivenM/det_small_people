from data_perp import CropDataset
from coaching import Coach
from models import VisTransformer
import argparse

import torch
from torchvision import transforms
from torch.utils.data import DataLoader


def main(config:dict):
    coach = Coach(config['name'])
    model = VisTransformer(config['num_blocks'])

    sample_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406,0.485, 0.456, 0.406, 0.406],
        std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.225] )
        ])
    target_transforms = transforms.Lambda(lambda x: torch.tensor([x], dtype=torch.float32)) 
    train_dataset = CropDataset('data/train.json', sample_transforms, target_transforms)
    val_dataset = CropDataset('data/val.json', sample_transforms, target_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    loss_fn = torch.nn.BCELoss()
    #optimizator = torch.optim.Adam(model.parameters(), lr=config['lr'])

    coach.fit(
        config['epoches'],
        model,
        loss_fn,
        config['lr'],
        train_dataloader,
        val_dataloader
        )
    

if __name__ == '__main__':
    config = {
        'name': 'exp2_obj_nblocks1_ep50',
        'num_blocks': 1,
        'epoches': 50,
        'lr': 0.001,
    }

    main(config)