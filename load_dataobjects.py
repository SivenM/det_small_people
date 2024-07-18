"""
Загружает объекты из датасетов в формат под обучение.
Сохранияет (кроп, класс) в pickle
"""

import os
import argparse
import utils
import pickle
from data_perp import CropsLoader


SAVE_DIR = "data"


def get_dirs(path:str) -> str:
    img_dir = os.path.join(path, 'images')
    ann_dir = os.path.join(path, 'ieos_vid_labels')
    return img_dir, ann_dir


def get_dataset(path:str) -> list:
    img_dir_path, ann_dir_path = get_dirs(path)
    crops_loader = CropsLoader(ann_dir_path, img_dir_path)
    data = crops_loader.load_data()
    return data


def to_pickle(sample:tuple, save_path:str) -> None:
    with open(save_path, 'wb') as f:
        pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)


def save(data:list, save_dir:str, dataset_name:str) -> None:
    for i, sample in enumerate(data):
        path = os.path.join(save_dir, dataset_name + '_' +str(i) + '.pickle')
        to_pickle(sample, path)


def main(dataset_path:str) -> None:
    dataset_name = dataset_path.split('/')[-1]
    data = get_dataset(dataset_path)
    assert len(data) > 0
    save(data, SAVE_DIR, dataset_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="load data for train")
    parser.add_argument('-i', '--inputs', type=str, help='input path')
    args = parser.parse_args()
    if args:
        main(args.inputs)