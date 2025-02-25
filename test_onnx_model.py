import os
import onnxruntime as ort
import numpy as np
import cv2
import csv
from tqdm import tqdm
from data_perp import CocoTestDataset
import utils
import visualizer as vis
from metrics import calc_det_metrics


class CSVMetricLogger:
    def __init__(self, save_dir_path):
        self.filepath = os.path.join(save_dir_path, 'metrics.csv')
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['idx', 'img_path', 'acc', 'precissoin', 'recall', 'iou'])

    def log(self, idx:int, img_path:str, acc:float, precission:float=None, recall:tuple=None, iou:str=None):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([idx, img_path, acc, precission, recall, iou])



class InferModel:
    def __init__(self, path:str):
        self.model_path = path
        self._load(path)

    def _load(self, path:str):
        self.model = ort.InferenceSession(path)
        self.in_img = self.model.get_inputs()[0].name
        self.in_shape = self.model.get_inputs()[1].name

    def preproc_image(self, image:np.ndarray) -> np.ndarray:
        """
        (H, W, C) -> (B, C, H, W)
        """
        if len(image.shape) == 2 or image.shape[-1] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = image.transpose((2, 0, 1))
        img = np.expand_dims(image, 0)
        img = img.astype(np.float32)
        img /= 255
        return img
    
    def __call__(self, image:np.ndarray, size:tuple) -> tuple[np.ndarray]:
        image = self.preproc_image(image)
        outputs = self.model.run(None, {self.in_img:image, self.in_shape:np.array([size])})
        return outputs
    

class Tester:
    def __init__(self, save_preds=True, save_vis=True, save_dir:str=None):
        self.save_preds = save_preds
        self.save_vis = save_vis
        self.save_preds = save_preds

        if save_dir is None:
            save_dir = 'test_results/test_onnx'
        
        self.save_dir = save_dir
        self.preds_path = os.path.join(save_dir, 'preds')
        self.images_dir = os.path.join(save_dir, 'images')
        self.results_path = os.path.join(save_dir, 'results.json')
        self.metric_logger = CSVMetricLogger(save_dir)
        self._create_dirs()

    def _create_dirs(self):
        utils.mkdir(self.save_dir)
        utils.mkdir(self.images_dir)
        utils.mkdir(self.preds_path)


    def save_img(self, img:np.ndarray, filename:str):
        save_path = os.path.join(self.images_dir, filename)
        cv2.imwrite(save_path, img)

    def run(self, model:ort.InferenceSession, dataset:CocoTestDataset, tr:float=0.1) ->None:
        preds = {}
        metrics = {
            'acc': [],
            'precission': [],
            'recall': [],
            'iou': []
        }
        for i, (image, targets, meta) in tqdm(enumerate(dataset)):
            labels, bboxes, scores = model(image)
            scores = scores[0]
            tr_scores = scores[scores > 0.01]
            tr_labels = labels[0][scores > 0.01]
            tr_bboxes = bboxes[0][scores > 0.01]
            acc, prec, recall, iou = calc_det_metrics(tr_bboxes, tr_labels, tr_scores, targets)
            metrics['acc'].append(acc)
            metrics['precission'].append(prec)
            metrics['recall'].append(recall)
            metrics['iou'].append(iou)
            preds[meta['file_name']] = {
                'bboxes': tr_bboxes, 
                'scores': tr_scores, 
                'targets': targets,
                'width': meta['width'],
                'height': meta['height']
                }
            self.metric_logger.log(i, meta['file_name'], acc, prec, recall, iou)
            draw_image = vis.show_img_pred(image, tr_bboxes)
            if self.save_vis:
                self.save_img(draw_image, meta['file_name'])
            
        m_acc = sum(metrics['acc']) / len(metrics['acc'])
        m_precission = sum(metrics['precission']) / len(metrics['precission'])
        m_recall = sum(metrics['recall']) / len(metrics['recall'])
        m_iou = sum(metrics['iou']) / len(metrics['iou'])
        results = {
            'num_images': len(dataset),
            'm_acc': m_acc,
            'm_precission': m_precission,
            'm_reacall': m_recall,
            'm_iou': m_iou
        }
        vis.bar(['acc', 'precission', 'recall', 'iou'], m_acc, m_precission, m_recall, m_iou)
        utils.save_json(preds, self.preds_path)
        utils.save_json(results, self.results_path)


def print_args(cfg:dict):
    print('cfg params:')
    for k, v in cfg.items():
        print(f'\t{k}: {v}')


def main(cfg:dict):
    print_args(cfg)
    model = InferModel(cfg['model'])
    dataset = CocoTestDataset(cfg['dataset'])
