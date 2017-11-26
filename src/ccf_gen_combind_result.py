#coding=utf-8

'''
生成结果CSV文件
'''

import numpy as np
import cv2
import csv
from tqdm import tqdm

mask1_pool = ['1_8bits_vegetation_predict.png','1_8bits_road_predict.png',
              '1_8bits_building_predict.png','1_8bits_water_predict.png']

mask2_pool = ['2_8bits_vegetation_predict.png','2_8bits_road_predict.png',
              '2_8bits_building_predict.png','2_8bits_water_predict.png']

mask3_pool = ['3_8bits_vegetation_predict.png','3_8bits_road_predict.png',
              '3_8bits_building_predict.png','3_8bits_water_predict.png']              

## 0:none  1:vegetation   2:road   3:building   4:water

#after mask combind
img_sets = ['final_1_8bits_predict.png','final_2_8bits_predict.png','final_3_8bits_predict.png']


def combind_all_mask():
    for mask_num in tqdm(range(3)):
        final_mask = np.zeros((5190,5204),np.uint8)#生成一个全黑全0图像
        #final_mask = cv2.imread('final_1_8bits_predict.png',0)
        
        if mask_num == 0:
            mask_pool = mask1_pool
        elif mask_num == 1:
            mask_pool = mask2_pool
        elif mask_num == 2:
            mask_pool = mask3_pool
        final_name = img_sets[mask_num]
        for idx,name in enumerate(mask_pool):
            img = cv2.imread(name,0)
            height,width = img.shape
            label_value = idx+1  #coressponding labels value
            for i in tqdm(range(height)):
                for j in range(width):
                    if final_mask[i,j] == 0 and img[i,j] == 255:
                        final_mask[i,j] = label_value
                        
        cv2.imwrite(final_name,final_mask)           
                
                
print 'combinding mask...'
combind_all_mask()                

print 'genrating result csv..'
for idx,name in enumerate(img_sets):
    img = cv2.imread(name,0)
    height,width = img.shape
    csv_name = ('%d.csv' % (idx+1))
    begin = ['ID','ret']
    with open(csv_name, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(begin)
        for i in tqdm(range(width)):
            for j in range(height):
                result=[]
                pixel = str(img[j,i])
                #result.append(pixel)
                str_idx = str(idx+1)
                result.append(str_idx)
                result.append(pixel)
                #print result
                writer.writerow(result)
        
    

