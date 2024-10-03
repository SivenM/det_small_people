import json
import os
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
import matplotlib.pylab as plt
from torchvision.io import read_image
from torchvision.ops.boxes import box_area
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


def show_sample(sample , target, rows=2, cols=5):
    fig = plt.figure(figsize=(8,8))
    if target == 1:
        label = 'human'
    else:
        label = 'bg'
    plt.title(label + f' | ')
    plt.axis('off')

    for i in range(len(sample)):
        img  = sample[i]
        fig.add_subplot(rows, cols, i+1)
        plt.title(str(img.shape[:2]))
        plt.axis('off')
        plt.imshow(img, cmap='gray')
    plt.show()


##############################33
# bbox utils


def to_corners(bboxes:Tensor):
    x, y, w, h = bboxes.unbind(-1)
    b_corners = [(x - 0.5 * w), (y - 0.5 * h),
         (x - 0.5 * w), (y - 0.5 * h)]
    return torch.stack(b_corners, dim=-1)


def to_xywh(bboxes:Tensor):
    x0, y0, x1, y1 = bboxes.unbind(-1)
    b_xywh = [(x0 + x1) / 2, (y0 + y1) / 2,
              (x1 - x0), (y1 - y0)]
    return torch.stack(b_xywh, dim=-1)


def get_iou(bboxes1:Tensor, bboxes2:Tensor):
    area1 = box_area(bboxes1)
    area2 = box_area(bboxes2)

    lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])
    rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:,:,0] * wh[:,:,1]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union


def generalized_iou(bboxes1:Tensor, bboxes2:Tensor):
    iou, union = get_iou(bboxes1, bboxes2)

    lt = torch.min(bboxes1[:, None, :2], bboxes2[:, :2])
    rb = torch.max(bboxes1[:, None, 2:], bboxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:,:,0] * wh[:,:,1]

    return iou - (inter - union) / inter