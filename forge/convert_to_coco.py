import os
import pickle
import json
import cv2
import numpy as np
from numpy import ndarray
from loguru import logger
from tqdm import tqdm
import argparse
import yaml


class Resizer:
    def __init__(self, size):
        self.size = size

    @staticmethod
    def _resize_coordinate(resized_img_side, coord, img_side):
        return resized_img_side * coord / img_side

    def resize_img(self, img):
        if self.size == [] or self.size is None:
            return img
        else:
            r_img = cv2.resize(img, self.size)
            if len(r_img.shape) == 2:
                r_img = np.expand_dims(r_img, axis=-1)
        return r_img
    
    def resize_coords(self, bboxes, img_size):
        if self.size == [] or self.size is None:
            return bboxes
        else:
            resized_bboxes = []
            for bbox in bboxes:
                resized_bboxes.append(
                    [
                        self._resize_coordinate(self.size[1], bbox[0], img_size[1]),
                        self._resize_coordinate(self.size[0], bbox[1], img_size[0]),
                        self._resize_coordinate(self.size[1], bbox[2], img_size[1]),
                        self._resize_coordinate(self.size[0], bbox[3], img_size[0]),
                    ]
                )
            return resized_bboxes


class AnnWorker:
    def __init__(self):
        self.ann = {
            "images": [], 
            "annotations": [], 
            "categories": [{'id': 1, "name": "person"}]}

    def add_ann_bboxes(self, bboxes:list) -> None:
        curr_num_ids = len(self.ann['annotations'])
        try:
            image_id = self.ann['images'][-1]['id']
            for bbox in bboxes:
                curr_num_ids += 1
                self.ann['annotations'].append(
                    {
                        "id": curr_num_ids,
                        "image_id": image_id,
                        "bbox": bbox, 
                        "category_id": 1, # в дальнейшем при появлении доп классов улучшить
                        "area": bbox[-1] * bbox[-2],
                        "iscrowd": 0,
                        "segmentation": None
                    }
                )
        except IndexError as e:
            logger.error(f'Сначало загрузи инфу об изображении!')

    def add_ann_image(self, img_name:str, img_size:tuple) -> None:
        num_imgs = len(self.ann['images'])
        self.ann['images'].append(
            {
                "id": num_imgs + 1,
                "file_name": img_name,
                "width": img_size[1],
                "height": img_size[0],
            }
        )

    def save_ann(self, save_dir):
        save_path = os.path.join(save_dir, 'annotations.json')
        with open(save_path, "w", encoding="utf8") as write_file:
            json.dump(self.ann, write_file, ensure_ascii=False)
            #logger.info(f'annotations saved in {save_path}')


class ImageWorker:
    def __init__(self, save_path, size:list=[]):
        self.save_path = save_path
        self.size = size
        self.resizer = Resizer(size=size)

    def to_rgb(self, sample:ndarray) -> ndarray:
        img = sample[:,:,0]
        return np.stack([img,img,img], axis=-1)

    def resize_data(self, sample:ndarray, bboxes:list):
        img_size = sample.shape[:-1]
        if img_size == self.size:
            return sample, bboxes
        else:
            sample = self.resizer.resize_img(sample)
            bboxes = self.resizer.resize_coords(bboxes, img_size)
            return sample, bboxes
    
    def save_img(self, sample:ndarray, name:str) -> tuple:
        img_name = name.split('.')[0] + '.jpg'
        img_size = sample.shape[:-1]
        save_path = os.path.join(self.save_path, img_name)
        try:
            cv2.imwrite(save_path, sample)
        except Exception as e:
            logger.error(e)
        return img_name, img_size


class FolderCreator:
    def __init__(self):
        pass
    
    def mkdir(self, path:str):
        if os.path.exists(path) == False:
            os.mkdir(path) 

    def create_dataset_folders(self, main_dir:str) -> None:
        if main_dir == None:
            return None, None
        images_dir = os.path.join(main_dir, 'images')
        self.mkdir(main_dir)
        self.mkdir(images_dir)
        return main_dir, images_dir

class ConvertToCoco:

    def __init__(self, save_path:str=None, resize:list=[], mode:str='rgb'):
        self.folder_creator = FolderCreator()
        save_path, save_images_path = self.folder_creator.create_dataset_folders(save_path)
        self.save_path = save_path
        self.img_worker = ImageWorker(save_images_path, resize)
        self.ann_worker = AnnWorker()
        self.mode = mode
        
    def _from_pickle(self, data_path:str) -> tuple:
        with open(data_path, "rb") as file:
            data = pickle.load(file)
        return data
    
    def convert_to_coco_bboxes(self, bboxes:list):
        """ XYWH -> XlYlWH """
        coco_bboxes = []
        for bbox in bboxes:
            coco_bboxes.append(
                [
                    int(bbox[0] - bbox[2] // 2),
                    int(bbox[1] - bbox[3] // 2),
                    bbox[2],
                    bbox[3],
                ]
            )
        return coco_bboxes
            
    def __call__(self, dataset_path:str) -> None:
        data_names = os.listdir(dataset_path)
        print(f'num samples: {len(data_names)}')
        for name in tqdm(data_names, desc='converting'):
            sample, bboxes = self._from_pickle(os.path.join(dataset_path, name))
            if sample.shape[-1] == 1 and self.mode == 'rgb':
                sample = self.img_worker.to_rgb(sample)
            sample, bboxes = self.img_worker.resize_data(sample, bboxes)
            bboxes = self.convert_to_coco_bboxes(bboxes)
            img_name, img_size = self.img_worker.save_img(sample, name)
            self.ann_worker.add_ann_image(img_name, img_size)
            self.ann_worker.add_ann_bboxes(bboxes)
        self.ann_worker.save_ann(self.save_path)
        print('Датасет конвертирован в coco формат!')


def main(config:dict):
    dataset_name = config['dataset_path'].split('/')[-1]
    print(f'dataset name: {dataset_name}')
    converter = ConvertToCoco(config['save_dir'], config['resize'], config['mode'])
    converter(config['dataset_path'])


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
