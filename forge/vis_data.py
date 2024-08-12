'''
Визуализирует data_human и data_bg
'''

import os
import sys
import matplotlib.pylab as plt
from tqdm import tqdm

sys.path.append('.')
import utils


def main(in_dir_list:str, out_dir_list:str) -> None:
    for i, in_dir in enumerate(in_dir_list):
        out_dir = out_dir_list[i]
        utils.mkdir(out_dir)
        for name in tqdm(os.listdir(in_dir), desc=f'{in_dir.split('/')[-1]}'):
            path = os.path.join(in_dir, name)
            sample, labels = utils.load_pickle(path)

            fig = plt.figure(figsize=(8,8))
            row, cols = 2, 5
            if labels == 1:
                    label = 'human'
            else:
                label = 'bg'
            plt.title(label)
            plt.axis('off')

            for i in range(len(sample)):
                img  = sample[i]
                fig.add_subplot(row, cols, i+1)
                plt.title(str(img.shape[:2]))
                plt.axis('off')
                plt.imshow(img, cmap='gray')
            save_path = os.path.join(out_dir, name.split('.')[0] + '.png')
            plt.savefig(save_path)
            fig.clf()
            plt.close()

if __name__ == '__main__':
    in_dir_list = [
        #'/home/max/ieos/small_obj/vid_pred/data/human_data',
        '/home/max/ieos/small_obj/vid_pred/data/data_bg'
    ]
    out_dir_list = [
        #'/home/max/ieos/small_obj/vid_pred/data/vis_human',
        '/home/max/ieos/small_obj/vid_pred/data/vis_bg'
    ]

    main(in_dir_list, out_dir_list)