import json
import os
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
import matplotlib.pylab as plt
from torchvision.io import read_image
import torchvision.transforms.functional as F  
import pickle


def read_img(path:str) -> Tensor:
    return read_image(path)


def show_imgs(imgs:Tensor | list) -> None:
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def load_json(path:str) -> list | dict:
    with open(path, 'rb') as read_file:
        ann = json.load(read_file)
    return ann


def save_json(data, save_path, desc=None):
    with open(save_path, "w", encoding="utf8") as write_file:
        json.dump(data, write_file, ensure_ascii=False)
        if desc:
            print(desc)


def save_pickle(sample:tuple, save_path:str) -> None:
    with open(save_path, 'wb') as f:
        pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def show_crop_sample(sample:ndarray|Tensor, label:ndarray|Tensor):
    #assert sample.shape[-1] == 3, f'shape must be (H, W, 3)'
    num_samples = len(sample)
    cols = 2
    raws = num_samples // 2
    fig = plt.figsize(8, 8)
    for i in range(num_samples):
        crop = sample[i]
        fig.add_subplot(raws, cols, i)
        plt.title(str(label[i]))
        plt.axis('off')
        if type(crop) == Tensor:
            plt.imshow(crop.squeeze(), cmap='gray')
        else:
            plt.imshow(crop, cmap='gray')


def mkdir(path:str) -> None:
    if os.path.exists(path) == False:
        os.mkdir(path)    
