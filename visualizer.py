import os
import matplotlib.pylab as plt
import cv2
from numpy import ndarray


def draw_bboxes(img_draw:ndarray, bboxes:ndarray, color:tuple=(0,0,255)) -> ndarray:
    for bbox in bboxes:
        img_draw = cv2.rectangle(img_draw, (int(bbox[2]), int(bbox[3])), (int(bbox[0]), int(bbox[1])),  color, 1)
    return img_draw


def show_img_pred(image:ndarray, preds:ndarray, targets:ndarray):
    img_draw = image.copy()
    img_draw = draw_bboxes(img_draw, preds)
    img_draw = draw_bboxes(img_draw, targets, (255,255,0))
    return img_draw


def bar(x:list, y:list, save=True, show=False):
    plt.bar(x, y, edgecolor='black', linewidth=1.2, label='Данные')
    plt.title('Результаты тестирования', fontsize=16, pad=20)
    plt.xlabel('Score', fontsize=14, labelpad=10)
    plt.ylabel('Metrics', fontsize=14, labelpad=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    if save:
        plt.savefig('overoll_metrics.png', format='png', dpi=300)
    if show:
        plt.tight_layout()
        plt.show()
