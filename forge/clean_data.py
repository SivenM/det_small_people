import sys
sys.path.append('.')

import utils
import os
from tqdm import tqdm
import argparse


class Cleaner:

    def __init__(self) -> None:
        pass

    def delete_file(self, file_path:str):
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Файл {file_path} успешно удален.")
            else:
                print(f"Файла {file_path} не существует.")
        except Exception as e:
            print(f"Произошла ошибка при удалении файла: {e}")

    def delete_path(self, idx:int, path_list:list) -> list:
        _ = path_list.pop(idx)
        return path_list

    def is_clean_sample(self, sample:list) -> bool:
        clean = True
        for obj in sample:
            size = obj.shape
            if size[0] == 0 or size[1] == 0:
                return False
        return clean

    def clean_objects_from_json(self, json_path:str):
        data = utils.load_json(json_path)
        clean_data = []
        del_count = 0
        for i in tqdm(range(len(data)), desc='data'):
            obj_path = data[i]
            if os.path.isfile(obj_path):
                loaded = utils.load_pickle(obj_path)
                sample, _ = loaded
                clean = self.is_clean_sample(sample)
                if clean:
                    clean_data.append(obj_path)
                else:
                    self.delete_file(obj_path)
                    del_count += 1
        print(f'\nКоличество "чистых" объектов: {len(clean_data)}')
        print(f'Количество удаленных объектов: {del_count}')

        print(f'Сохраняем чистые данные в {json_path}')
        utils.save_json(clean_data, json_path)
        print('Done!')


def main(json_path:str) -> None:
    cleaner = Cleaner()
    cleaner.clean_objects_from_json(json_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputs', type=str, help='путь до json файла')

    args = parser.parse_args()
    if args:
        main(args.inputs)
    else:
        print('А где путь до файла?')