#coding=utf-8

import cv2
from tqdm import tqdm
import os

## 0:none  1:vegetation   2:road   3:building   4:water

img_sets = ['1_class_8bits','2_class_8bits']
mask_labels = ['none','vegetation','road','building','water']


def gen_one_mask(mask_all,value):
    mask_one = mask_all.copy()
    v_mask_one = mask_all.copy()
    h,w = mask_all.shape
    for i in tqdm(range(h)):
        for j in range(w):
            if mask_all[i,j] == value:
                mask_one[i,j] = 1
                v_mask_one[i,j] = 255
            else:
                mask_one[i,j] = 0
                v_mask_one[i,j] = 0
    return mask_one,v_mask_one
    
def gen_mask(prefix):
    for i in tqdm(range(len(img_sets))):
        src_name = img_sets[i]+'.png'
        im = cv2.imread(src_name,0)
        for label_idx in tqdm(range(len(mask_labels))):
            if label_idx == 0:
                continue                      
            mask_name = prefix+'/'+img_sets[i]+'_'+mask_labels[label_idx]+'_'+'mask.tif'
            v_mask_name = prefix+'/'+'visualized_'+img_sets[i]+'_'+mask_labels[label_idx]+'_'+'mask.png'           
            mask,v_mask = gen_one_mask(im,label_idx)   #v_mask: visualized mask
            cv2.imwrite(mask_name,mask)
            cv2.imwrite(v_mask_name,v_mask)
            

PREFIX = './mask_pool'
if not os.path.exists(PREFIX):
    os.makedirs(PREFIX)
    
gen_mask(PREFIX)
