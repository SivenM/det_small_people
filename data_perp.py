import os
import numpy as np
from numpy import ndarray
import utils
import torch
import cv2
#from torch import Tensor
#from torchvision.io import read_image
#import torchvision.transforms as transforms
from tqdm import tqdm


class CropsLoader:

    def __init__(self, ann_dir, img_dir, chunk_size=10) -> None:
        self.ann_dir = ann_dir
        self.ann_names = os.listdir(self.ann_dir)
        self.img_dir = img_dir
        self.chunk_size = chunk_size
        #self.obj_data = self._load_data()
    
    def load_img(self, frame_num:int) -> ndarray:
        img_name = str(frame_num-1) + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        img =  cv2.imread(img_path)
        assert type(img) == ndarray, f'frame num: {frame_num}\npath: {img_path}'
        return img

    def load_ann(self, ann_name:str) -> dict|list:
        ann_path = os.path.join(self.ann_dir, ann_name)
        return utils.load_json(ann_path)

    def crop(self, img:ndarray, crop:list) -> ndarray:
        return img[crop[1]:crop[3], crop[0]:crop[2], :]

    def to_chunks(self, crops:list) -> list:
        chunk_list = []
        num_chunks = len(crops) // self.chunk_size
        if num_chunks > 0:
            for i in range(num_chunks):
                chunk = crops[i*self.chunk_size: (i+1) * self.chunk_size]
                chunk_list.append(chunk)
            return chunk_list
        else:
            return crops

    def load_data(self) -> list[tuple]:
        data = []
        for i in tqdm(range(len(self.ann_names)), desc='anns'):
            ann_name = self.ann_names[i]
            ann = self.load_ann(ann_name)
            if ann['label'] == 'human':
                crop_list = []
                for j, frame_num in enumerate(ann['frames']):
                    img = self.load_img(frame_num)
                    coords = ann['coords'][j]
                    crop = self.crop(img, coords)
                    crop_list.append(crop)    
                chunks = self.to_chunks(crop_list)
                data.append((chunks, np.ones((len(chunks),))))
        return data

