import os
import numpy as np
from numpy import ndarray
import utils
import torch
import cv2
import pickle
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
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

    def get_crops(self, ann:dict) -> list:
        crop_list = []
        for j, frame_num in enumerate(ann['frames']):
            img = self.load_img(frame_num)
            coords = ann['coords'][j]
            crop = self.crop(img, coords)
            crop_list.append(crop)
        return crop_list
    
    def exctruct_obj(self, ann:dict) -> tuple:
        """
        Возвращяет объект в виде кропов кусками размером chunk_size
        и номером класса (человек = 1)
        """
        crop_list = self.get_crops(ann)    
        chunks = self.to_chunks(crop_list)
        cls = np.ones((len(chunks),))
        return (chunks, cls)

    def load_data(self) -> list[tuple]:
        data = []
        for i in tqdm(range(len(self.ann_names)), desc='anns'):
            ann_name = self.ann_names[i]
            ann = self.load_ann(ann_name)
            if ann['label'] == 'human':
                obj = self.exctruct_obj(ann)
                data.append(obj)
        return data



class CropDataset(Dataset):
    
    def __init__(self, data_dir, transform=None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.data_names = os.listdir(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data_names)
    
    def from_pickle(self, data_path:str):
        with open(data_path, "rb") as file:
            data = pickle.load(file)
        return data

    def __getitem__(self, index:int):
        data_path = os.path.join(self.data_dir, self.data_names[index])
        sample, label = self.from_pickle(data_path)
        return sample, label
    
    