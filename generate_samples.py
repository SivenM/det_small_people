"""
Скрипт генерирует сэмплы для обучения и тестирования из видео
"""

import os
from data_perp import SampleGenerator
import json
import argparse


def mkdir(path:str):
    if os.path.exists(path) == False:
        os.mkdir(path)


def run(dataset_path:str, save_path:str, frame_rate:int=10, indent:int=0, mode='seq'):
    imgs_dir = os.path.join(dataset_path, 'images')
    ann_dir = os.path.join(dataset_path, 'ieos_labels')
    mkdir(save_path)

    sample_generator = SampleGenerator(imgs_dir, ann_dir, frame_rate, indent, mode)
    sample_generator.generate(save_path)
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Скрипт для генерации сэмплов')
    parser.add_argument('-c', '--cfg', type=argparse.FileType('r'), default=None, help='конфиг для настройки запуска скрипта')
    args = parser.parse_args()
    if args.cfg:
        cfg = json.load(args.cfg)
        run(
            cfg['dataset'],
            cfg['save_path'],
            cfg['frame_rate'],
            cfg['indent'],
            cfg['mode']
        )