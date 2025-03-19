import os
import matplotlib.pylab as plt
import cv2
from numpy import ndarray
import utils


def draw_bboxes(img_draw:ndarray, bboxes:ndarray, color:tuple=(0,0,255), thicknes:int=1) -> ndarray:
    for bbox in bboxes:
        img_draw = cv2.rectangle(img_draw, (int(bbox[2]), int(bbox[3])), (int(bbox[0]), int(bbox[1])),  color, thicknes)
    return img_draw


def show_img_pred(image:ndarray, preds:ndarray, targets:ndarray=None, color=(0,0,255), thickness=1):
    img_draw = image.copy()
    img_draw = draw_bboxes(img_draw, preds, color, thickness)
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