#coding=utf-8

import cv2
from tqdm import tqdm

img_sets = ['1_class_8bits','2_class_8bits']

for i in tqdm(range(len(img_sets))):
    src_name = img_sets[i]+'.png'
    tif_name = img_sets[i]+'.tif'
    im = cv2.imread(src_name)
    cv2.imwrite(tif_name,im)
