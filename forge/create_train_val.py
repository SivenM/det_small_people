import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

SAVE_DIR = "/home/max/ieos/small_obj/vid_pred/data"

def save_json(data, save_path, desc=None):
    with open(save_path, "w", encoding="utf8") as write_file:
        json.dump(data, write_file, ensure_ascii=False)
        if desc:
            print(desc)


def run(human_dir_path:str, bg_dir_path:str, save_dir:str=SAVE_DIR, val_size=0.33) -> None:
    print("run splitting...")
    human_paths = list(map(lambda x: os.path.join(human_dir_path, x), sorted(os.listdir(human_dir_path))))
    bg_paths = list(map(lambda x: os.path.join(bg_dir_path, x), sorted(os.listdir(bg_dir_path))))
    human_y = np.ones(len(human_paths))
    bg_y = np.zeros(len(bg_paths))

    x_human_train, x_human_val, _, _ = train_test_split(human_paths, human_y, test_size=val_size, random_state=42)
    x_bg_train, x_bg_val, _, _ = train_test_split(bg_paths, bg_y, test_size=val_size, random_state=42)

    print(f"num  train human: {len(x_human_train)}")
    print(f"num  val human: {len(x_human_val)}")
    print(f"num train bg: {len(x_bg_train)}")
    print(f"num val bg: {len(x_bg_val)}\n")

    x_train = x_human_train + x_bg_train
    x_val = x_human_val + x_bg_val

    print("saving...")
    train_save_path = os.path.join(save_dir, 'train.json')
    val_save_path = os.path.join(save_dir, 'val.json')
    save_json(x_train, train_save_path)
    save_json(x_val, val_save_path)
    print("Done!")


if __name__ == '__main__':
    human_dir_path =  "data/human_data"
    bg_dif_path = "data/data_bg"
    run(human_dir_path, bg_dif_path)