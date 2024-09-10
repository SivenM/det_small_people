import os
import sys
import shutil
from sklearn.model_selection import train_test_split
import argparse
import json

sys.path.append('.')
import utils

def copy_samples(paths:dict):
    for src, dst in zip(paths['src'], paths['dst']):
        shutil.copy(src, dst)


def run(in_path:str, out_path_train:str, out_path_test:str):
    print(f'in path: {in_path}')
    print(f'out paths: \n{out_path_train}\n{out_path_test}')
    utils.mkdir(out_path_train)
    utils.mkdir(out_path_test)
    samples = sorted(os.listdir(in_path))
    print(f'\nnum samples: {len(samples)}')

    train, test = train_test_split(samples, test_size=0.2)
    print(f'num train: {len(train)}')
    print(f'num test: {len(test)}')
    print('\ncopying...')
    train_paths = {
        'src': list(map(lambda x: os.path.join(in_path, x), train)),
        'dst': list(map(lambda x: os.path.join(out_path_train, x), train))
    }
    test_paths = {
        'src': list(map(lambda x: os.path.join(in_path, x), test)),
        'dst': list(map(lambda x: os.path.join(out_path_test, x), test))
    }

    for paths in [train_paths, test_paths]:
        copy_samples(paths)
    print('Done!')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('сплит сэмплов на обучающую и тестовую выборки')
    parser.add_argument('-c', '--cfg', type=argparse.FileType('r'), default=None, help='конфиг для настройки запуска скрипта')
    args = parser.parse_args()
    if args.cfg:
        cfg = json.load(args.cfg)
        run(
            cfg['in_path'],
            cfg['out_path_train'],
            cfg['out_path_test']
        )