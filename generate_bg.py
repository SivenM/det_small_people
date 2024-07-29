import os
import argparse
from data_perp import BgGenerator
from loguru import logger


SAVE_DIR = "data_bg"


def get_dirs(path:str) -> str:
    img_dir = os.path.join(path, 'images')
    ann_dir = os.path.join(path, 'ieos_labels')
    return img_dir, ann_dir

#@logger.trace
def main(dataset_path:str, num_samples:int, img_size:tuple, range_width:tuple, range_height:tuple) -> None:
    img_dir, ann_dir = get_dirs(dataset_path)
    logger.info(f'img dir: {img_dir}\nann_dir: {ann_dir}')
    bg_gen = BgGenerator(
        img_dir,
        ann_dir,
        range_height,
        range_width,
        10,
        img_size
    )
    logger.info("bg gen created")
    bg_gen.generate(num_samples, save_dir=SAVE_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Программа для генерации сэмплов фона")
    parser.add_argument('-i', '--inputs', type=str, help='Путь до датасета')
    parser.add_argument('-n', '--num', type=int, default=2000, help='Количество bg сэмплов')
    parser.add_argument('-is', '--img_size', nargs='+', default=(480, 640), help='размеры изображения (H, W)')
    parser.add_argument('-w', '--range_width', nargs='+', default=(10, 50), help='ширина')
    parser.add_argument('-he', '--range_height', nargs='+', default=(10, 50), help='высота')
    args = parser.parse_args()
    logger.info("parsed")
    main(
        args.inputs, 
        args.num, 
        args.img_size, 
        args.range_width, 
        args.range_height
        )