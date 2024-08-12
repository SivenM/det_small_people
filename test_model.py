from data_perp import TestCropDataset
import models
import utils

import os
import csv
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torcheval.metrics.functional import binary_accuracy
import matplotlib.pylab as plt
from tqdm import tqdm
import argparse


class CSVLogger:
    def __init__(self, dir_path):
        self.filepath = os.path.join(dir_path, 'results.csv')
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['idx', 'pred', 'gt', 'acc', 'height', 'width', 'img_path'])

    def log(self, idx:int, pred:float, gt:float, acc:float, size:tuple, img_path:str):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([idx, pred, gt, acc, size[0], size[1], img_path])


class ObjTester:

    def __init__(self, dataset:Dataset, save_dir=None, vis=True) -> None:
        self.dataset = dataset
        self.vis = vis
        self.save_dir = save_dir
        utils.mkdir(self.save_dir)
        self.results_dir = os.path.join(save_dir, 'results')
        utils.mkdir(self.results_dir)
        self.pred_logger = CSVLogger(save_dir)

    def show_pred(self, sample, target, pred, s_size=None, idx=None, show=False):
        fig = plt.figure(figsize=(10,8))
        row, cols = 2, 5
        if target == 1:
            label = 'human'
        else:
            label = 'bg'
        if s_size:
            plt.title(label + f' | pred: {pred:.2f}\n sample size: {s_size}')
        else:
            plt.title(label + f' | pred: {pred:.2f}')

        plt.axis('off')

        for i in range(len(sample)):
            img  = sample[i]
            fig.add_subplot(row, cols, i+1)
            #plt.title(str(img.size[:2]))
            plt.axis('off')
            plt.imshow(img, cmap='gray')
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(self.results_dir, str(idx)+'.jpg'), bbox_inches='tight', pad_inches=0.1)
        fig.clf()
        plt.close()
    
    def stat(self, sample, pred, gt, size, idx, img_path):
        acc = binary_accuracy(pred, gt)
        self.show_pred(sample, gt, pred, size, idx)
        self.pred_logger.log(
            idx,
            pred,
            gt,
            acc,
            size,
            img_path
        )

    def test_model(self, model):
        for i, (sample, gt, meta) in tqdm(enumerate(self.dataset), desc='testing'):
            pred = model(sample.unsqueeze(0))
            acc = binary_accuracy(pred.unsqueeze(0), gt)
            self.show_pred(sample, gt, pred, meta['size'], i)
            self.pred_logger.log(
                i,
                pred.item(),
                gt.item(),
                acc.item(),
                meta['size'],
                meta['path']
            )


class TestModel:

    def __init__(self, model_path:str, num_blocks:int=1, device='cpu') -> None:
        self.model_path = model_path
        self.device = device
        self.model = models.VisTransformer(num_blocks)
        self._load_model()

    def _load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, x:torch.Tensor):
        return self.model(x)
    

def load_dataset(dataset_path:str):
    mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406,0.485, 0.456, 0.406, 0.406]
    std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.225]
    sample_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
        std=std)
        ])
    target_transforms = transforms.Lambda(lambda x: torch.tensor([x], dtype=torch.float32)) 
    val_dataset = TestCropDataset(dataset_path, sample_transforms, target_transforms)
    return val_dataset


def main(config:dict) -> None:
    test_model = TestModel(config['model'], config['num_blocks'])
    test_dataset = load_dataset(config['dataset'])
    tester = ObjTester(test_dataset, config['save_dir'])

    tester.test_model(test_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=argparse.FileType('r'), default=None, help='конфиг входных данных')
    args = parser.parse_args()
    if args.config:
        config = json.load(args.config)
        main(config)