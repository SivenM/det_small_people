from data_perp import TestCropDataset, DETRDataset, DetrLocDatasetTest
from models import transformer, detr_zoo
import utils

import os
import csv
import yaml
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torcheval.metrics.functional import binary_accuracy
import matplotlib.pylab as plt
from tqdm import tqdm
import cv2
import numpy as np
import argparse


class CSVLogger:
    def __init__(self, dir_path):
        self.filepath = os.path.join(dir_path, 'results.csv')
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['idx', 'pred', 'gt', 'acc', 'height', 'width', 'img_path'])

    def log(self, idx:int, pred:float, gt:float, acc:float=None, size:tuple=None, img_path:str=None):
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


class DetrLocTester:
    
    def __init__(self, dataset:Dataset, save_dir:str=None, vis:bool=True) -> None:
        self.dataset = dataset
        self.vis = vis
        self.save_dir = save_dir
        utils.mkdir(self.save_dir)
        self.results_dir = os.path.join(save_dir, 'results')
        utils.mkdir(self.results_dir)
        self.pred_logger = CSVLogger(save_dir)

    def _denorm(self, t_bboxes:Tensor, img_size:torch.Size) -> Tensor:
        if t_bboxes.shape[0] > 0:
            t_bboxes[:, 0] *= img_size[1]
            t_bboxes[:, 1] *= img_size[0]
            t_bboxes[:, 2] *= img_size[1]
            t_bboxes[:, 3] *= img_size[0]
        return t_bboxes

    def test_model(self, model):
        #for i, (sample, sample_np, gt, gt_norm, meta) in tqdm(enumerate(self.dataset), desc='testing'):
        sample, sample_np, gt, gt_norm, meta = self.dataset[10]
        pred = model(sample.unsqueeze(0))
        print(meta['size'])
        bboxes = self._denorm(pred[0].clone(), meta['size'])
        print(f'gt norm:')
        for box in gt_norm:
            print(box)
        print(f'\npreds:')
        for box in pred:
            print(box)
        print(f'gt:')
        for box in gt:
            print(box)
        print(f'\ndenorm preds:')
        for box in bboxes:
            print(box)

            
        color_gt = (255,0,0)    
        color_denorm = (0,255,0)    
        img = sample_np[-1].copy()
        img_gt = np.stack([img,img,img], axis=-1)
        img_dn = np.stack([img,img,img], axis=-1)
        bboxes_corner = utils.to_corners(bboxes) 
        gt_corners = utils.to_corners(gt)
        for box in gt_corners:
            img_gt = cv2.rectangle(img_gt, (int(box[2]), int(box[3])), (int(box[0]), int(box[1])),  color_gt)
        for box in bboxes_corner:
            img_dn = cv2.rectangle(img_dn, (int(box[2]), int(box[3])), (int(box[0]), int(box[1])),  color_denorm)
        out_img = np.concatenate([img_gt, img_dn], axis=0)
        plt.imshow(out_img)
        plt.show()
            #for box in bboxes:
            #    print(box.sum())
            #mask = bboxes >= 1
            #bboxes_squeezed = bboxes[mask]
            #print(f'\n{bboxes.shape=}')
            #print(f'{bboxes_squeezed.shape=}')
            #break
            #self.show_pred(sample, gt, pred, meta['size'], i)
            #self.pred_logger.log(
            #    i,
            #    pred.item(),
            #    gt.item(),
            #    size=meta['size'],
            #    img_path=meta['path']
            #)


class DetrTester:
    pass


class TestModel:

    def __init__(self, model_path:str, model_type:str, model_params:dict, device='cpu') -> None:
        self.model_path = model_path
        self.device = device
        if model_type == 'vit':
            self.model = transformer.VisTransformer(model_params['encoder_blocks'])
        elif model_type == 'detr_loc':
            self.model = detr_zoo.DetrLoc(
                    model_params['num_encoder_blocks'],
                    model_params['num_decoder_blocks'],
                    model_params['num_queries'],
                    model_params['num_cls'],
                    model_params['emb_dim'],
                    model_params['img_size'],
                    model_params['num_imgs'],
                    model_params['patch_size']
                    )
        elif model_type == 'detr':
            self.model = detr_zoo.DETR(
                    model_params['num_encoder_blocks'],
                    model_params['num_decoder_blocks'],
                    model_params['num_queries'],
                    model_params['num_cls'],
                    model_params['emb_dim'],
                    model_params['img_size'],
                    model_params['num_imgs'],
                    model_params['patch_size']
                    )
        else:
            raise 'Wrong model type. Only: vit, detr, detr_loc'
        self._load_model()

    def _load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, x:torch.Tensor):
        return self.model(x)
    

def load_dataset(dataset_path:str, model_type:str):
    if model_type == 'vit':
        mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406,0.485, 0.456, 0.406, 0.406]
        std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.225]
        sample_transforms = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=mean,
            std=std)
            ])
        target_transforms = v2.Lambda(lambda x: torch.tensor([x], dtype=torch.float32)) 
        dataset = TestCropDataset(dataset_path, sample_transforms, target_transforms)
    
    sample_transforms = v2.Compose([
        #transforms.ToTensor(),
        v2.Lambda(lambda x: torch.tensor(x)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5,0.5, 0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5,0.5, 0.5, 0.5, 0.5] )
        ])
    if model_type == 'detr':
        dataset = DETRDataset(
            dataset_path, 
            norm=True, 
            transform=sample_transforms
            )
    elif model_type == 'detr_loc':
        dataset = DetrLocDatasetTest(
            dataset_path, 
            norm=True, 
            transform=sample_transforms
            )
    else:
        raise 'Wrong model type. Only: vit, detr, detr_loc'
    return dataset


def create_tester(dataset:Dataset, model_type:str, save_dir:str):
    if model_type == 'vit':
        return ObjTester(dataset, save_dir)
    elif model_type == 'detr_loc':
        return DetrLocTester(dataset, save_dir)
    elif model_type == 'detr':
        pass
    else:
        raise 'Wrong model type. Only: vit, detr, detr_loc'


def main(config:dict) -> None:
    test_model = TestModel(config['model_path'], config['model_type'], config['model_params'])
    test_dataset = load_dataset(config['dataset'], config['model_type'])
    tester = create_tester(test_dataset, config['model_type'], config['save_dir'])

    tester.test_model(test_model)
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=argparse.FileType('r'), default=None, help='конфиг входных данных')
    args = parser.parse_args()
    if args.config:
        try:
            config = yaml.safe_load(args.config)
            main(config)
        except FileNotFoundError:
            print(f'Файл {args.config} не найден')
        except yaml.YAMLError as exc:
            print(f'Ошибка при чтении YAML файла: {exc}')