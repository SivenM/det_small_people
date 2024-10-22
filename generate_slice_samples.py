"""
Скрипт генерирует сэмплы для обучения и тестирования из слайсов видео
"""

import os
from data_perp import SampleGenerator
import yaml
import argparse


def mkdir(path:str):
    if os.path.exists(path) == False:
        os.mkdir(path)
        print(f'dir {path} created!')


def run(cfg:dict):
    mkdir(cfg['save_path'])
    print(f'start generating...')
    for d_slice in cfg['data_slices']:
        imgs_dir = os.path.join(d_slice['path'], 'images')
        ann_dir = os.path.join(d_slice['path'], 'ieos_labels')
        sample_generator = SampleGenerator(imgs_dir, ann_dir, cfg['frame_rate'], cfg['indent'])
        sample_generator.generate_from_slices(d_slice['slices'], cfg['save_path'])
        print(f'samples from {d_slice['path'].split('/')[-1]} generated in {cfg['save_path']}')
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Скрипт для генерации сэмплов')
    parser.add_argument('-c', '--cfg', type=argparse.FileType('r'), default=None, help='конфиг для настройки запуска скрипта')
    args = parser.parse_args()
    if args.cfg:
        cfg = yaml.safe_load(args.cfg)
        run(cfg)