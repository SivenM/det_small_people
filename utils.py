import json
import torch
from torch import Tensor
import numpy as np
import matplotlib.pylab as plt
from torchvision.io import read_image
import torchvision.transforms.functional as F  


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

            