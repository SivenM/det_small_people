"""
Делает из изображений видео
"""
import os
import cv2
import numpy as np
from tqdm import tqdm


def get_range(range_frames:list) -> range:
    if len(range_frames) == 0:
        return range(len(os.listdir(main_dir))-1)
    else:
        return range(range_frames[0], range_frames[1])


def create_vids(main_dir:str, range_frames:list, save_dir:str, prefix:str=''):
    if os.path.exists(save_dir)  == False:
        os.mkdir(save_dir)

    vid_name = main_dir.split('/')[-1] + '_' + prefix
    vid_name_path = os.path.join(save_dir, vid_name  + '.mp4')
    
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    vid = cv2.VideoWriter(vid_name_path, codec, 30, (640, 480))
    
    iter_frames = get_range(range_frames)    
    for num_frame in tqdm(iter_frames, desc='converting'):
        img_name = str(num_frame) + '.jpg'
        img_path = os.path.join(main_dir, img_name)
        vid.write(cv2.resize(cv2.imread(img_path), (640, 480)))
    vid.release()


if __name__ == '__main__':
    main_dir = "results/pred_stroika_vid3_tr_00_7/window_vid_4"
    range_frames = [1800, 5300]
    save_dir = "results/vids/"
    prefix = '01'

    create_vids(main_dir, range_frames, save_dir, prefix)