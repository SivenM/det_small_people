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

    sample_transforms = transforms.Compose([transforms.ToTensor()])
    target_transforms = transforms.Lambda(lambda x: torch.tensor([x], dtype=torch.float32)) 
    train_dataset = CropDataset('data/train.json', sample_transforms, target_transforms)
    val_dataset = CropDataset('data/val.json', sample_transforms, target_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizator = torch.optim.Adam(model.parameters(), lr=config['lr'])

    coach.fit(
        config['epoches'],
        model,
        loss_fn,
        optimizator,
        train_dataloader,
        val_dataloader
        )
    

if __name__ == '__main__':
    config = {
        'name': 'exp1_obj_numblock_1',
        'num_blocks': 1,
        'epoches': 100,
        'lr': 0.001,

    }