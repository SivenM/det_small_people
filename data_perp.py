import os
import numpy as np
from numpy import ndarray
import random
import utils
import cv2
import pickle
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
import torch.nn.functional as F
import torch
from torch import Tensor
from tqdm import tqdm
from loguru import logger


class Loader:

    def __init__(self, img_dir, ann_dir, mode) -> None:
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        if mode == 'seq':
            self.ann_names = np.arange(1, len(os.listdir(self.ann_dir))-3)
        elif mode == 'classic': #only frame_rate == 1 !
            self.ann_names = os.listdir(self.ann_dir)

    def load_img(self, frame_num:int) -> ndarray:
        if self.mode == 'seq':
            img_name = str(frame_num-1) + '.jpg'
        else:
            img_name = str(frame_num) + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        img =  cv2.imread(img_path)
        assert type(img) == ndarray, f'frame num: {frame_num}\npath: {img_path}'
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=-1)
        return img

    def load_ann(self, ann_name:str) -> dict|list:
        ann_path = os.path.join(self.ann_dir, ann_name)
        return utils.load_json(ann_path)


class CropsLoader(Loader):

    def __init__(self, img_dir:str, ann_dir:str, chunk_size=10) -> None:
        super().__init__(img_dir, ann_dir)
        print(f'num objects: {len(self.ann_names)}')
        self.chunk_size = chunk_size
        self.ann_names = os.listdir(ann_dir)

    def crop(self, img:ndarray, crop:list) -> ndarray:
        return img[crop[1]:crop[3], crop[0]:crop[2], :]

    def to_chunks(self, crops:list) -> list:
        chunk_list = []
        num_chunks = len(crops) // self.chunk_size
        if num_chunks > 0:
            for i in range(num_chunks):
                chunk = crops[i*self.chunk_size: (i+1) * self.chunk_size]
                chunk_list.append((chunk, 1))
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
    
    def exctruct_obj(self, ann:dict) -> list[tuple]:
        """
        Возвращяет список размером chunk_size с чанками 
        и номером класса (человек = 1) соответствующим им
        """
        crop_list = self.get_crops(ann)    
        chunks = self.to_chunks(crop_list)
        return chunks

    def load_data(self) -> list[tuple]:
        data = []
        for ann_name in tqdm(self.ann_names, desc='anns'):
            ann = self.load_ann(ann_name)
            if ann['label'].lower() == 'human':
                obj = self.exctruct_obj(ann)
                data += obj
        return data


class BgGenerator(Loader):

    def __init__(self, img_dir:str=None, ann_dir:str=None, range_hieght:tuple|list=(), range_width:tuple|list=(), chunk_size:int=10, img_size:tuple=(480, 640)) -> None:
        super().__init__(img_dir, ann_dir)
        self.dataset_name = ann_dir.split('/')[-2]
        self.chunk_size = chunk_size
        self.img_size = img_size
        self.range_height = range_hieght
        self.range_width = range_width

    def get_random_ann(self):
        ann = {}
        while True:
            idx = random.randint(1, len(self.ann_names)-3)
            ann_name = str(self.ann_names[idx]) + '.json'
            path = os.path.join(self.ann_dir, ann_name)
            ann = self.load_ann(path)
            if ann['num_objects'] > 0 and int(ann['name']) < len(self.ann_names)-3:
                    return ann

    def get_coords(self, ann:dict) -> ndarray:
        coords = []
        for obj in ann['objects']:
            coords.append(obj['corner_bboxes'])
        return np.array(coords)

    def create_bg(self) -> ndarray:
        top_x = random.randint(0, self.img_size[1])
        top_y = random.randint(0, self.img_size[0])
        running = True
        while running:
            width = random.randint(self.range_width[0], self.range_width[1])
            height = random.randint(self.range_height[0], self.range_height[1])
            if width < height:
                running = False
        bg_coord = np.array([top_x, top_y, top_x + width, top_y + height])
        return bg_coord

    def __iou_base(self, target:ndarray, crops:ndarray):
        lu = np.maximum(crops[:, :2], target[:2])
        rd = np.minimum(crops[:, 2:], target[2:])
        intersection = np.maximum(0.0, rd - lu)
        intersection_area = intersection[:, 0] * intersection[:, 1]
        target_area = (target[2] - target[0]) * (target[3] - target[1])
        crops_area = (crops[:, 2] - crops[:, 0]) * (crops[:, 3] - crops[:, 1])
        union_area = np.maximum(crops_area + target_area - intersection_area, 1e-8)
        return np.clip(intersection_area / union_area, 0.0, 1.0)
    
    def check_overlap(self, target:ndarray, crops:ndarray):
        lu = np.maximum(crops[:, :2], target[:2])
        rd = np.minimum(crops[:, 2:], target[2:])
        intersection = np.maximum(0.0, rd - lu)
        intersection_area = intersection[:, 0] * intersection[:, 1]
        target_area = (target[2] - target[0]) * (target[3] - target[1])
        return np.clip(intersection_area / target_area, 0.0, 1.0)

    def load_imgs_bg(self, start_frame_name:int) -> ndarray:
        imgs = [self.load_img(i) for i in range(start_frame_name, start_frame_name + self.chunk_size)]
        imgs = np.array(imgs)
        return imgs

    def crop_bg(self, bg_coord:list, imgs:ndarray|list) -> tuple:
        label = 0
        if type(imgs) == list:
            imgs = np.array(imgs)
        
        bg = imgs[:, bg_coord[1]:bg_coord[3], bg_coord[0]:bg_coord[2], :]
        #out = list(map(lambda x: (x[0], x[1]), zip(bg, labels)))
        return (bg, label)

    def update_bg(self, obj_coords:ndarray) -> ndarray:
        while True:
            bg_coords = self.create_bg()
            iou = self.check_overlap(bg_coords, obj_coords)
            if len(np.where(iou < 0.3)[0]) > 0:
                return bg_coords

    def save(self, data:tuple, num:int, save_dir:str):
        save_path = os.path.join(save_dir, self.dataset_name + '_' + str(num) + '.pickle')
        utils.save_pickle(data, save_path)

    def generate(self, num:int, save_dir) -> None:
        print('generating')
        for i in tqdm(range(num), desc='bg samples'):
            ann = self.get_random_ann()
            if int(ann['name']) > self.ann_names[-1]:
                print(int(ann['name']))
                continue
            obj_coords = self.get_coords(ann)
            bg_coords = self.create_bg()
            iou = self.check_overlap(bg_coords, obj_coords)
            if len(np.where(iou > 0.3)[0]) > 0:
                bg_coords = self.update_bg(obj_coords)
            imgs = self.load_imgs_bg(int(ann['name']))
            bg_crops_labels = self.crop_bg(bg_coords, imgs)
            self.save(bg_crops_labels, i, save_dir)
        print('done!')
        print(f'saved in {save_dir}')


class SampleGenerator(Loader):
    """
    Генерирует сэмплы для обучения DETR
    """
    def __init__(self, img_dir:str, ann_dir:str, frame_rate:int=10, indent:int=0, mode='seq') -> None:
        """
        img_dir: путь до директории изображений
        ann_dir: путь до директории аннотации
        frame_rate: кол-во кадров для одного сэмпла
        indent: отступ влево привзятии кадров из  ieos датасета
        от 0 до frame_rate 
        """
        super().__init__(img_dir, ann_dir, mode)
        self.frame_rate = frame_rate
        self.indent = indent
        self.dataset_name = img_dir.split('/')[-2]
        self.mode = mode

    def get_ann(self, idx:int):
        if self.mode == 'seq':
            ann_name = str(self.ann_names[idx]) + '.json'
        elif self.mode == 'classic':
            ann_name = self.ann_names[idx]
        path = os.path.join(self.ann_dir, ann_name)
        ann = self.load_ann(path)
        return ann

    def load_imgs_seq(self, start_frame_name:int) -> ndarray:
        imgs = [self.load_img(i) for i in range(start_frame_name, start_frame_name + self.frame_rate)]
        imgs = np.array(imgs)
        return np.squeeze(imgs)

    def to_targets(self, ann:dict) -> list:
        bboxes = []
        if len(ann['objects']) > 0:
            for obj in ann['objects']:
                bboxes.append(obj['xywh_bboxes'])
        return bboxes

    def gen_sample(self, idx:int) -> tuple:
        start_ann = self.get_ann(idx)
        if self.frame_rate > 1:
            end_ann = self.get_ann(idx + self.frame_rate)
            imgs = self.load_imgs_seq(int(start_ann['name']))
            targets = self.to_targets(end_ann)
        else:
            if self.mode == 'classic':
                img_idx = int(start_ann['name'].split('.')[0])
            else:
                img_idx = int(start_ann['name'])
            imgs = self.load_img(img_idx)
            targets = self.to_targets(start_ann)
        return (imgs, targets)
        

    def save(self, data:tuple, num:int, save_dir:str):
        save_path = os.path.join(save_dir, self.dataset_name + '_' + str(num) + '_' + str(self.indent) + '.pickle')
        utils.save_pickle(data, save_path)

    def _gen(self, iterible):
        pass

    def generate(self, save_dir:str=None):
        num_imgs = len(self.ann_names)
        num_iters = num_imgs // self.frame_rate
        iterate = (x * self.frame_rate  - self.indent for x in range(num_iters))
        for i, idx in tqdm(enumerate(iterate)):
            if i == 0 and self.indent > 0:
                continue
            sample = self.gen_sample(idx)
            if save_dir:
                self.save(sample, i, save_dir)
            else:
                print(f'imgs shape: {sample[0].shape}')

    def generate_from_slices(self, slice_list:list, save_dir:str=None):
        for slice in slice_list:
            start_imgs_idx = np.arange(slice[0]- self.indent, slice[1] - self.indent, self.frame_rate)
            for i, idx in enumerate(start_imgs_idx):
                if i == 0 and self.indent > 0:
                    continue
                sample = self.gen_sample(idx)
                if save_dir:
                    self.save(sample, idx, save_dir)
                else:
                    print(f'imgs shape: {sample[0].shape}')


class Pad():

    def __init__(self, patch_size:tuple, ):
        self.patch_size = patch_size

    def get_pad_h(self, h:int):
        assert h <= self.patch_size[0], f"Height ({h}) must be lower than patch_size {self.patch_size[0]}"
        if h == self.patch_size[0]:
            return (0,0)
        else:
            diff = self.patch_size[-1] - h
            if diff % 2 == 0:
                return (diff // 2, diff // 2)
            else:
                return (diff // 2, diff // 2 + 1)
        
    def get_pad_w(self, w:int):
        assert w <= self.patch_size[-1], f"Height ({w}) must be lower than patch_size {self.patch_size[-1]}"
        if w == self.patch_size[0]:
            return (0,0)
        else:
            diff = self.patch_size[-1] - w
            if diff % 2 == 0:
                return (diff // 2, diff // 2)
            else:
                return (diff // 2, diff // 2 + 1)
    
    def get_pad(self, img:Tensor):
        _, h, w = img.shape

        pad_h = self.get_pad_h(h)
        pad_w = self.get_pad_w(w)
        padded_img = F.pad(img, (pad_w[0], pad_w[1], pad_h[0], pad_h[1]), 'constant', 0)
        return padded_img


class CropDataset(Dataset):
    
    def __init__(self, json_path, transform=None, target_transforms=None, pad_size:tuple=(50,50)) -> None:
        super().__init__()
        self.pad_size = pad_size
        if pad_size:
            self.pad = Pad(pad_size)
        else:
            self.pad = None
        self.path_list = self._load(json_path)
        self.transform = transform
        self.target_transforms = target_transforms

    def __len__(self):
        return len(self.path_list)

    def _load(self, path):
        return utils.load_json(path)
    
    def from_pickle(self, data_path:str):
        with open(data_path, "rb") as file:
            data = pickle.load(file)
        return data

    def resize_crops(self, sample:list) -> list:
        h, w = sample[0].shape[:2]
        if h > self.pad_size[0]:
            h = self.pad_size[0]
        if w > self.pad_size[1]:
            w = self.pad_size[1]
        
        out = []
        for i in range(0, len(sample)):
            img = sample[i]
            assert type(img) == np.ndarray
            assert len(img.shape) == 3
            assert img.shape[-1] == 1
            #logger.info(img.shape)
            resized = cv2.resize(img, (w, h))
            out.append(np.expand_dims(resized, axis=-1))
        return np.concatenate(out, axis=2)
    
    def prepare_sample(self, sample:list) -> np.ndarray:
        
        #    sample_np = np.stack(sample)
        sample_np = self.resize_crops(sample)
        return sample_np

    def __getitem__(self, index:int):
        data_path = self.path_list[index]
        sample, label = self.from_pickle(data_path)
        assert len(sample) == 10
        sample = self.prepare_sample(sample)
        if self.transform:
            sample = self.transform(sample)
        if self.target_transforms:
            label = self.target_transforms(label)
        if self.pad:
            sample = self.pad.get_pad(sample)
        return sample, label    
    

class TestCropDataset(CropDataset):
    
    def __init__(self, json_path, transform=None, target_transforms=None, pad_size:tuple=(50,50)) -> None:
        super().__init__(json_path, transform, target_transforms, pad_size)

    def __getitem__(self, index:int):
        data_path = self.path_list[index]
        sample, label = self.from_pickle(data_path)
        assert len(sample) == 10
        sample = self.prepare_sample(sample)
        sample_size = sample.shape[:2]
        if self.transform:
            sample = self.transform(sample)
        if self.target_transforms:
            label = self.target_transforms(label)
        if self.pad:
            sample = self.pad.get_pad(sample)
        return sample, label, {'size':sample_size, 'path': data_path}    
    

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
                        int(self._resize_coordinate(self.size[1], bbox[0], img_size[1])),
                        int(self._resize_coordinate(self.size[0], bbox[1], img_size[0])),
                        int(self._resize_coordinate(self.size[1], bbox[2], img_size[1])),
                        int(self._resize_coordinate(self.size[0], bbox[3], img_size[0])),
                    ]
                )
            return resized_bboxes



class DETRDataset(Dataset):

    def __init__(self, dir_path:str, norm=True, size=[], transform=None, target_transforms=None, mode:str='base') -> None:
        super().__init__()
        self.dir_path = dir_path
        self.norm = norm
        self.samples = os.listdir(dir_path)
        self.transform = transform
        self.target_transforms = target_transforms
        self.mode = mode #base, rgb, time
        self.resizer = Resizer(size)

    def __len__(self):
        return len(self.samples)
    
    def _from_pickle(self, data_path:str):
        with open(data_path, "rb") as file:
            data = pickle.load(file)
        return data

    def normalize(self, t_bboxes:Tensor, img_size:tuple):
        """
        Нормирование координат рамок от 0 до 1
        """   
        if t_bboxes.shape[0] > 0:
            t_bboxes[:, 0] /= img_size[1]
            t_bboxes[:, 1] /= img_size[0]
            t_bboxes[:, 2] /= img_size[1]
            t_bboxes[:, 3] /= img_size[0]
        return t_bboxes
    
    def to_rgb(self, sample):
        return np.concatenate([sample, sample, sample], axis=-1)

    def _create_targets(self, bboxes:list, img_size:tuple) -> Tensor:
        t_bboxes = torch.tensor(bboxes, dtype=torch.float32)
        if self.norm:
            t_bboxes = self.normalize(t_bboxes, img_size)
        t_labels = torch.ones(len(bboxes), dtype=torch.long)
        if self.target_transforms:
            t_bboxes = self.target_transforms(t_bboxes)
        return {'bbox': t_bboxes, 'labels': t_labels}

    def __getitem__(self, index:int):
        sample_name = self.samples[index]
        sample, bboxes = self._from_pickle(os.path.join(self.dir_path, sample_name))
        if sample.shape[-1] == 1:
            img_size = sample.shape[:-1]
        else:
            img_size = sample.shape[1:]
        sample = self.resizer.resize_img(sample)
        bboxes = self.resizer.resize_coords(bboxes, img_size)
        targets = self._create_targets(bboxes, img_size)
        if self.mode == 'rgb':
            sample = self.to_rgb(sample)
        if self.transform:
            sample = self.transform(np.array(sample))
            if self.mode == 'time':
                sample = sample.unsqueeze(1)
        return sample, targets
    

class DetrLocDataset(DETRDataset):

    def __init__(self, dir_path: str, norm=True, transform=None, target_transforms=None, rgb:bool=False) -> None:
        super().__init__(dir_path, norm, transform, target_transforms)
        self.norm = norm
        self.rgb=rgb

    def __getitem__(self, index: int):
        sample_name = self.samples[index]
        sample, bboxes = self._from_pickle(os.path.join(self.dir_path, sample_name))
        if self.rgb:
            if type(sample) == np.ndarray:
                sample = np.concatenate([sample, sample, sample], axis=2)
            else:
                raise 'sample not ndarray'
        if self.norm:
            bboxes = self.normalize(torch.tensor(bboxes, dtype=torch.float32), sample.shape[1:])
        else:
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
        if self.transform:
            sample = self.transform(np.array(sample))
        return sample, bboxes


class LocDataLoader(DETRDataset):
    def __init__(self, dir_path: str, max_bboxes:int, norm=True, transform=None, target_transforms=None) -> None:
        super().__init__(dir_path, norm, transform, target_transforms)
        self.max_bboxes = max_bboxes

    def to_targets(self, bboxes:torch.Tensor) -> torch.Tensor:
        targets = torch.zeros(self.max_bboxes, 4)
        num_bboxes = bboxes.shape[0]
        if num_bboxes == 0:
            return targets
        else:
            targets[:num_bboxes, :] = bboxes
            return targets

    def __getitem__(self, index: int):
        sample_name = self.samples[index]
        sample, bboxes = self._from_pickle(os.path.join(self.dir_path, sample_name))
        if self.norm:
            bboxes = self.normalize(torch.tensor(bboxes, dtype=torch.float32), sample.shape[1:])
        else:
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
        targets = self.to_targets(bboxes)
        if self.transform:
            sample = self.transform(np.array(sample))
        return sample, targets


#### Dataloaders for tests ##########
###############################################################################################################


class DetrLocDatasetTest(DetrLocDataset):

    def __init__(self, dir_path: str, norm=True, transform=None, target_transforms=None) -> None:
        super().__init__(dir_path, norm, transform, target_transforms)

    def __getitem__(self, index: int):
        sample_name = self.samples[index]
        sample, bboxes = self._from_pickle(os.path.join(self.dir_path, sample_name))
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        bboxes_norm = self.normalize(torch.tensor(bboxes, dtype=torch.float32), sample.shape[1:])
        if self.transform:
            t_sample = self.transform(np.array(sample))
        return t_sample, sample, bboxes, bboxes_norm, {'path': sample_name, 'size': sample.shape[1:]}


class DETRDatasetTest(DETRDataset):

    def __init__(self, dir_path: str, norm=True, size=[], transform=None, target_transforms=None, mode:str='base') -> None:
        super().__init__(dir_path, norm, size, transform, target_transforms, mode=mode)

    def _create_targets(self, bboxes:list, img_size:tuple) -> Tensor:
        t_bboxes = torch.tensor(bboxes, dtype=torch.float32)
        t_bboxes = self.normalize(t_bboxes, img_size)
        t_labels = torch.ones(len(bboxes), dtype=torch.long)
        if self.target_transforms:
            t_bboxes = self.target_transforms(t_bboxes)
        return t_bboxes, t_labels

    def __getitem__(self, index:int):
        sample_name = self.samples[index]
        sample, bboxes = self._from_pickle(os.path.join(self.dir_path, sample_name))
        if sample.shape[-1] == 1:
            img_size = sample.shape[:-1]
        else:
            img_size = sample.shape[1:]
        sample = self.resizer.resize_img(sample)
        bboxes = self.resizer.resize_coords(bboxes, img_size)
        t_bboxes, t_labels = self._create_targets(bboxes, img_size)
        if self.mode == 'rgb':
            sample = self.to_rgb(sample)
        if self.transform:
            t_sample = self.transform(np.array(sample))
        return sample, t_sample, bboxes, t_bboxes, t_labels, {'path': sample_name, 'size': sample.shape}


class CocoTestDataset(Dataset):
    def __init__(self, path, size=[]):
        super().__init__()
        self.path = path
        self.resizer = Resizer(size)
        self.images_dir = os.path.join(path, 'images')
        self.ann_path = os.path.join(path, 'annotations.json')
        self.ann = utils.load_json(self.ann_path)

    def __len__(self):
        return len(self.ann['images'])

    def load_img(self, file_name:str) -> np.ndarray:
        """
        return (H, W, 3)
        """
        path = os.path.join(self.images_dir, file_name)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def convert_to_xyxy(self, bbox):
        return [
                bbox[0],
                bbox[1],
                bbox[0] + bbox[2],
                bbox[1] + bbox[3],
                ]

    def get_targets(self, image_id:str) -> np.ndarray: 
        """
        return bboxes (N, 4)
        """
        bboxes = []
        for ann in self.ann['annotations']:
            if ann['image_id'] == image_id:
                bboxes.append(self.convert_to_xyxy(ann['bbox']))
        return np.array(bboxes)

    def __getitem__(self, index):
        img_data = self.ann['images'][index]
        img = self.load_img(img_data['file_name'])
        img_size = img.shape[:-1]
        targets = self.get_targets(img_data['id'])
        img = self.resizer.resize_img(img)
        targets = np.array(self.resizer.resize_coords(targets, img_size))
        return img, targets, {'file_name': img_data['file_name'], 'width': img.shape[1], 'height': img.shape[0]}

###############################################################################################################


class SeqCls(Dataset):

    def __init__(self, dir_path, sample_transform=None, target_transform=None, meta:bool=False) -> None:
        super().__init__()
        self.dir_path = dir_path
        self.samples = os.listdir(dir_path)
        self.sample_transform = sample_transform
        self.target_transform = target_transform
        self.meta = meta

    def __len__(self):
        return len(self.samples)
    
    def _from_pickle(self, data_path:str):
        with open(data_path, "rb") as file:
            data = pickle.load(file)
        return data

    def get_target(self, bboxes:list) -> Tensor:
        if len(bboxes) == 0:
            return 0.
        else:
            return 1.

    def __getitem__(self, index) -> tuple:
        sample_name = self.samples[index]
        sample, bboxes = self._from_pickle(os.path.join(self.dir_path, sample_name))
        #t_sample = torch.tensor(sample)
        target = self.get_target(bboxes)
        if self.sample_transform:
            sample = self.sample_transform(sample)
        if self.target_transform:
            target = self.target_transform(target)
        if self.meta:
            return sample, target, {'name': sample_name, 'size': sample.shape}
        else:
            return sample, target
