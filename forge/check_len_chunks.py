import sys
sys.path.append('.')

import utils
import os


OBJ_DIR = 'data/human_data'

def main():
    path_list = list(map(lambda x: os.path.join(OBJ_DIR, x), os.listdir(OBJ_DIR)))
    anomaly_count = 0
    for path in path_list:
        obj = utils.load_pickle(path)
        if len(obj) > 2 or len(obj) < 2:
            anomaly_count +=1
            print(f'{path}')


if __name__ == '__main__':
    main()