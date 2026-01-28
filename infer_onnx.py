import os
from pathlib import Path
import onnxruntime as ort
import numpy as np
import cv2
#from tqdm import tqdm
import matplotlib.pylab as plt
import visualizer as vis
import argparse
import yaml
import utils
from tqdm import tqdm
#import torchvision.transforms as T
#from loguru import logger


IMG_FORMATS = {'.jpg', '.png', '.jpeg'}


class Resizer:
    def __init__(self, size):
        self.size = size

    @staticmethod
    def _resize_coordinate(resized_img_side, coord, img_side, reverse=False):
        if reverse:
            out = img_side * coord / resized_img_side
            if out < 0:
                return 3
            return out
        else:
            return resized_img_side * coord / img_side

    def resize_img(self, img):
        if self.size == [] or self.size is None:
            return img
        else:
            r_img = cv2.resize(img, self.size)
            if len(r_img.shape) == 2:
                r_img = np.expand_dims(r_img, axis=-1)
            return r_img
    
    def resize_coords(self, bboxes, img_size, reverse=False) -> list:
        if self.size == [] or self.size is None:
            return bboxes
        else:
            resized_bboxes = []
            for bbox in bboxes:
                resized_bboxes.append(
                    [
                        int(self._resize_coordinate(self.size[1], bbox[0], img_size[1], reverse)),
                        int(self._resize_coordinate(self.size[0], bbox[1], img_size[0], reverse)),
                        int(self._resize_coordinate(self.size[1], bbox[2], img_size[1], reverse)),
                        int(self._resize_coordinate(self.size[0], bbox[3], img_size[0], reverse)),
                    ]
                )
            return resized_bboxes
        

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
        img_shape = image.shape
        h, w, _ = img_shape
        orig_size = np.array([[w, h]], dtype=np.int64)
        if img_shape[0] != self.size[0][0] or img_shape[1] != self.size[0][1]:
            image = cv2.resize(image, self.size[0], interpolation=cv2.INTER_LINEAR)
        img = image.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0) 
        return img, orig_size
        #print(f'image shape orig: {image.shape}')
        #orig_size = torch.tensor([image.shape[1], image.shape[0]])[None]
        #transforms = T.Compose([
        #    T.ToPILImage(),
        #    T.Resize((640, 640)),
        #    T.ToTensor(),
        #])
        #img = transforms(image)[None]
        #print(f'img data: {img.shape}')
        #print(f'orig_size: {orig_size} | {orig_size.shape}')
        #return img.numpy(), orig_size.numpy()

    def nms(self, bboxes, scores, threshold=0.5):
        if len(bboxes) == 0:
            return np.array([], dtype=int)
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            inds = np.where(iou <= threshold)[0]
            if inds.size == 0:
                break
            order = order[inds + 1] if inds.size > 0 else np.array([], dtype=int)  # +1, потому что order[0] уже обработан
        return np.array(keep)

    def __call__(self, image:np.ndarray, tr:list=[0, 0.1]) -> tuple[np.ndarray]:
        image, img_shape = self.preproc_image(image)
        labels, bboxes, scores = self.model.run(None, {self.in_img:image, self.in_shape:img_shape})
        scores = scores[0]
        labels = labels[0]
        bboxes = bboxes[0]

        #l_mask = labels != 1
        #scores = scores[l_mask]
        #bboxes = bboxes[l_mask]
        #labels = labels[l_mask]
        mask = (scores > tr[0]) & (scores < tr[1])
        tr_scores = scores[mask]
        tr_bboxes = bboxes[mask]
        tr_labels = labels[mask]
        #print(f'{tr_scores.shape=}')
        #print(f'{tr_bboxes.shape=}')
        keep_indices = self.nms(tr_bboxes, tr_scores, threshold=0.3)
#
        ## Отфильтрованные bounding boxes
        tr_bboxes = tr_bboxes[keep_indices]
        tr_scores = tr_scores[keep_indices]
        tr_labels = tr_labels[keep_indices]
        ##tr_bboxes = tr_bboxes[scores > tr[1]]
        #if image.shape[2:] != img_shape[:-1]:
        #    tr_bboxes = self.resizer.resize_coords(tr_bboxes, img_shape[0], True)
        return tr_bboxes, tr_scores, labels


def load_image(path:str) -> np.ndarray:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #h, w = img.shape[:2]
    #orig_size = np.array([[w, h]], dtype=np.int64)
    #im_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    #
    ## Нормализация и преобразование в формат [1, C, H, W]
    #im_data = im_resized.astype(np.float32) / 255.0  # Нормализация
    #im_data = np.transpose(im_data, (2, 0, 1))  # HWC -> CHW
    #im_data = np.expand_dims(im_data, axis=0) 
    return img, None


def save_img(img:np.ndarray, path:str):
    cv2.imwrite(path, img)


def pred_vid(model:InferModel, data_path, save_dir, tr, color, thic=3): 
    vid_name = data_path.split("/")[-1]
    print(f'vid: {vid_name}')
    print(f'infer runing...')
    cap = cv2.VideoCapture(data_path)
    if cap.isOpened() == False:
        print('Error')
    img_dir_path = os.path.join(save_dir, 'images')
    utils.mkdir(img_dir_path)
    count = 0
    pred_list = []
    while True:
        count += 1
        ret, frame = cap.read()
        if not ret:
            break
        bboxes, scores, labels = model(frame, tr)
        draw_image = vis.show_img_pred(frame, bboxes, conf=scores, labels=labels, color=color, thickness=thic)
        save_path = os.path.join(img_dir_path, str(count) + '.jpg')
        save_img(draw_image, save_path)
        pred_list.append(
            {
                'image_id': count,
                'img_size': frame.shape,
                'bboxes': bboxes.tolist(),
                'scores': scores.tolist(),
                'labels': labels.tolist()
            }
        )
    result_path = os.path.join(save_dir, f'{vid_name.split(".")[0]}_result.json')
    utils.save_json(pred_list, result_path)
    print('done')


def is_vid(path:str) -> str:
    vid_formats = ['mp4', 'webm']
    name = path.split('/')[-1]
    if name.split('.')[-1] in vid_formats:
        return 'vid'
    else:
        return 'folder'


def main(cfg:dict):
    model = InferModel(cfg['model_path'], cfg['in_size'])
    print(f'mode: {config['mode']}')
    if cfg['mode'] == 'img':
        source_path = Path(cfg['data_path'])
        if source_path.is_file():
            img, orig_size = load_image(cfg['data_path'])
            #img = cv2.resize(img, [640, 640])
            bboxes, scores, labels = model(img, cfg['tr'])
            draw_image = vis.show_img_pred(img, bboxes, color=cfg['color'], thickness=cfg['thic'])
            print(draw_image.shape)
            plt.imshow(draw_image)
            plt.show()
        elif source_path.is_dir():
            utils.mkdir(cfg['save_path'])
            for i, img_path in tqdm(enumerate(source_path.iterdir())):
                if img_path.suffix.lower() in IMG_FORMATS:
                    img, orig_size = load_image(str(img_path))
                    bboxes, scores, labels = model(img, cfg['tr'])
                    draw_image = vis.show_img_pred(img, bboxes, conf=scores, color=cfg['color'], thickness=cfg['thic'])           
                    save_path = os.path.join(cfg['save_path'], f'{i}.jpg')
                    save_img(draw_image, save_path)
            print('done!')
        else:
            print(f'Wrong data_path: {source_path}')
    elif cfg['mode'] == 'vid':
        utils.mkdir(cfg['save_path'])
        data_type = is_vid(cfg['data_path'])
        print(f'data type: {data_type}')
        if data_type == 'vid':
            pred_vid(model, 
                     config['data_path'], config['save_path'], 
                     config['tr'], config['color'],config['thic']
                     )
        else:
            vid_names = os.listdir(cfg['data_path'])
            print(f'num vids: {len(vid_names)}')
            for vid_name in vid_names:
                data_path = os.path.join(cfg['data_path'], vid_name)
                save_path = os.path.join(cfg['save_path'], vid_name)
                utils.mkdir(save_path)
                pred_vid(model, 
                         data_path, save_path, 
                         config['tr'], config['color'], config['thic']
                         )
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