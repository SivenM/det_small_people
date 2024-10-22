import os
import sys
sys.path.append('.')
import utils
import shutil
import argparse


def run(names:dict, sample_dir:str, human_save_path:str, bg_save_path:str):
    utils.mkdir(human_save_path)
    utils.mkdir(bg_save_path)
    for tg, name_list in names.items():
        print(f'type: {tg}\nnum:{len(name_list)}')
        print('copying...')
        for name in name_list:
            src_path = os.path.join(sample_dir, name)
            if tg == 'pos':
                dst_path = os.path.join(human_save_path, name)
            else:
                dst_path = os.path.join(bg_save_path, name)
            shutil.copyfile(src_path, dst_path)
        print(f'Done!')

if __name__ == '__main__':
    names_path = "data/seq_cls/pos_neg_names.json"
    sample_dir = "data/seq_cls/samples"
    human_save_path = "data/seq_cls/human/samples"
    bg_save_path = "data/seq_cls/bg/samples"
    names = utils.load_json(names_path)
    run(names, sample_dir, human_save_path, bg_save_path)