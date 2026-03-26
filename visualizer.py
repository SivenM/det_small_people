import os
import matplotlib.pylab as plt
import cv2
from numpy import ndarray
import utils
import numpy as np
import math


def draw_bboxes(
        img_draw:ndarray, 
        bboxes:ndarray, 
        conf:ndarray=None, 
        labels:ndarray=None,
        color:tuple=(0,0,255), 
        thicknes:int=1) -> ndarray:
    
    for i, bbox in enumerate(bboxes):
        if conf is not None:
            score = conf[i]
            text = f"{score:.2f}"
            img_draw = cv2.putText(img_draw, text, (int(bbox[0]) + 3, int(bbox[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                1,
                cv2.LINE_AA)
        if labels is not None:
            label = labels[i]
            text = f"{label}"
            img_draw = cv2.putText(img_draw, text, (int(bbox[2]) - 30, int(bbox[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                3,
                cv2.LINE_AA)
        img_draw = cv2.rectangle(img_draw, (int(bbox[2]), int(bbox[3])), (int(bbox[0]), int(bbox[1])),  color, thicknes)
    return img_draw


def show_img_pred(image:ndarray, preds:ndarray, conf:ndarray=None, labels:ndarray=None, targets:ndarray=None, color=(0,0,255), thickness=1):
    img_draw = image.copy()
    img_draw = draw_bboxes(img_draw, preds, conf, labels, color, thickness)
    if targets is not None:
        img_draw = draw_bboxes(img_draw, targets, (255,255,0))
    return img_draw


def bar(x:list, y:list, save_path:str=None, show=False):
    plt.bar(x, y, edgecolor='black', linewidth=1.2, label='Данные')
    plt.title('Результаты тестирования', fontsize=16, pad=20)
    plt.xlabel('Score', fontsize=14, labelpad=10)
    plt.ylabel('Metrics', fontsize=14, labelpad=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    if save_path:
        plt.savefig(save_path, format='png', dpi=500)
    if show:
        plt.tight_layout()
        plt.show()


def scatter_metric(x_data:list, y_metric:list, x_name:str, metric_name:str, save_path:str=None, show=False):
    assert len(x_data) == len(y_metric)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_metric, c='blue', alpha=0.7, edgecolors='w', s=100)
    plt.title(f'Зависимость {metric_name} от {x_name} bounding box', fontsize=14)
    plt.xlabel(f'{x_name}', fontsize=12)
    plt.ylabel(f'{metric_name}', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Добавляем цветовую шкалу (опционально)
    #cbar = plt.colorbar()
    #cbar.set_label('Интенсивность', rotation=270, labelpad=15)

    if save_path:
        plt.savefig(save_path, format='png', dpi=500)
    
    if show:
        plt.tight_layout()
        plt.show()


def get_pad_size(samples:list[np.ndarray]) -> tuple:
    sizes = [x.shape for x in samples]
    heights = [size[0] for size in sizes]
    weights = [size[1] for size in sizes]
    return (max(weights), max(heights))


def put_samples_in_image(samples:list[np.ndarray], image:np.ndarray, num_columns:int) -> np.ndarray:
    j = 0
    for i, sample in enumerate(samples):
        h, w, _ = sample.shape
        if i == 0:
            image[0:h, 0:w, :] = sample
        else:
            val = i % num_columns
            if val == 0:
                j += 1
            image[j*h:j*h+h, val*w:val*w+w, :] = sample
    return image


def get_vis(samples:list[np.ndarray], num_columns:int=10) -> np.ndarray:
    num_images = len(samples)
    pad_size = get_pad_size(samples)
    padding = utils.ZeroPad(pad_size)
    padded_samples = []
    for sample in samples:
        padded_samples.append(padding(sample))
    num_raws = math.ceil(num_images / num_columns)
    vis_image = np.zeros(
        shape=(
            num_raws * padded_samples[0].shape[0], 
            num_columns * padded_samples[0].shape[1], 
            padded_samples[0].shape[2]
            ), 
        dtype=padded_samples[0].dtype
        )
    vis_image = put_samples_in_image(padded_samples, vis_image, num_columns)
    return vis_image