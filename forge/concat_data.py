import os
import sys
sys.path.append('.')
import utils
import shutil


def concat(data_path1:str, data_path2:str, out_dir):
    utils.mkdir(out_dir)
    for data in [data_path1, data_path2]:
        for name in os.listdir(data):
            src = os.path.join(data, name)
            dst = os.path.join(out_dir, name)
            shutil.copy2(src, dst)


if __name__ == '__main__':
    data1 = "/home/max/ieos/small_obj/vid_pred/data/seq_cls/human/test"
    data2 = "/home/max/ieos/small_obj/vid_pred/data/seq_cls/bg/test"
    save_path = "/home/max/ieos/small_obj/vid_pred/data/seq_cls/test"
    concat(data1, data2, save_path)