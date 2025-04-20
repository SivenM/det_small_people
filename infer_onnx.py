import os
import onnxruntime as ort
import numpy as np
import cv2
from data_perp import Resizer
#from tqdm import tqdm
import matplotlib.pylab as plt
import visualizer as vis
import argparse
import yaml
import utils
from loguru import logger



class InferModel:
    def __init__(self, path:str, size):
        self.model_path = path
        self.size = np.array([size])
        self._load(path)
        self.resizer = Resizer(size)

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
        img_shape = image.shape
        if img_shape[0] != self.size[0][0] or img_shape[1] != self.size[0][1]:
            image = cv2.resize(image, self.size[0])
        img = image.transpose((2, 0, 1))
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32)
        img /= 255
        return img, img_shape

    def nms(self, bboxes, scores, threshold=0.5):
        """
        Реализация Non-Maximum Suppression (NMS) для подавления лишних bounding boxes.

        Параметры:
            bboxes (np.ndarray): Массив bounding boxes в формате [x1, y1, x2, y2].
            scores (np.ndarray): Массив scores (уверенность модели) для каждого bounding box.
            threshold (float): Порог IoU для подавления (обычно от 0.3 до 0.5).

        Возвращает:
            np.ndarray: Индексы bounding boxes, которые остались после NMS.
        """
        if len(bboxes) == 0:
            return np.array([], dtype=int)

        # Координаты bounding boxes
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        # Вычисляем площадь bounding boxes
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Сортируем bounding boxes по убыванию scores
        order = scores.argsort()[::-1]

        # Список для хранения индексов выбранных bounding boxes
        keep = []

        while order.size > 0:
            # Выбираем bounding box с наибольшим score
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            # Вычисляем IoU с остальными bounding boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # Вычисляем ширину и высоту пересечения
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            intersection = w * h

            # Вычисляем IoU
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            # Оставляем только те bounding boxes, у которых IoU меньше порога
            inds = np.where(iou <= threshold)[0]
            if inds.size == 0:
                break
            order = order[inds + 1] if inds.size > 0 else np.array([], dtype=int)  # +1, потому что order[0] уже обработан

        return np.array(keep)

    def __call__(self, image:np.ndarray, tr:list=[0, 0.1]) -> tuple[np.ndarray]:
        image, img_shape = self.preproc_image(image)
        _, bboxes, scores = self.model.run(None, {self.in_img:image, self.in_shape:self.size})
        scores = scores[0]
        mask = (scores > tr[0]) & (scores < tr[1])
        tr_scores = scores[mask]
        tr_bboxes = bboxes[0][mask]
        #print(f'{tr_scores.shape=}')
        #print(f'{tr_bboxes.shape=}')
        keep_indices = self.nms(tr_bboxes, tr_scores, threshold=0.3)

        # Отфильтрованные bounding boxes
        tr_bboxes = tr_bboxes[keep_indices]
        tr_scores = tr_scores[keep_indices]
        #tr_bboxes = tr_bboxes[scores > tr[1]]
        if image.shape[2:] != img_shape[:-1]:
            tr_bboxes = self.resizer.resize_coords(tr_bboxes, img_shape, True)
        return tr_bboxes, tr_scores


def load_image(path:str) -> np.ndarray:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, [img.shape[0], img.shape[1]]


def save_img(img:np.ndarray, path:str):
    cv2.imwrite(path, img)


def pred_vid(model:InferModel, cfg:dict): 
    print(f'vid: {cfg["data_path"].split("/")[-1]}')
    print(f'infer runing...')
    cap = cv2.VideoCapture(cfg['data_path'])
    if cap.isOpened() == False:
        print('Error')
    count = 0
    while True:
        count += 1
        ret, frame = cap.read()
        if not ret:
            break
        bboxes, scores = model(frame, cfg['tr'])
        draw_image = vis.show_img_pred(frame, bboxes, color=cfg['color'], thickness=cfg['thic'])
        save_path = os.path.join(cfg['save_path'], str(count) + '.jpg')
        save_img(draw_image, save_path)
    print('done')


def main(cfg:dict):
    model = InferModel(cfg['model_path'], cfg['in_size'])
    utils.mkdir(cfg['save_path'])
    if cfg['mode'] == 'img':
        img, img_size = load_image(cfg['data_path'])
        #img = cv2.resize(img, [640, 640])
        print(img.shape)
        bboxes, scores = model(img, cfg['tr'])
        draw_image = vis.show_img_pred(img, bboxes, color=cfg['color'], thickness=cfg['thic'])
        print(draw_image.shape)
        plt.imshow(draw_image)
        plt.show()
        #save_img(draw_image, cfg['save_path'])
    elif cfg['mode'] == 'vid':
        pred_vid(model, cfg)
    else:
        print(f'wrong mode: {cfg["mode"]}')
    
    #plt.imshow(draw_image)
    #plt.show()

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