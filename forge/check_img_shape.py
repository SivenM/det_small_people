import os
import cv2


img_dir = "/home/max/ieos/small_obj/vid_pred/data/simple_detr_dataset/coco_train_1_frame_v2/images"
img_names = os.listdir(img_dir)
for name in img_names:
    img = cv2.imread(os.path.join(img_dir, name))
    size = img.shape
    if size[:2] != (224,224):
        print(size)