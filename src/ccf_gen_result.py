#coding=utf-8

'''
生成结果CSV文件
'''

import numpy as np
import cv2
import csv
from tqdm import tqdm

img_sets = ['1_8bits_predict.png','2_8bits_predict.png','3_8bits_predict.png']

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
                pixel = str((img[j,i]/255)*3)
                #result.append(pixel)
                str_idx = str(idx+1)
                result.append(str_idx)
                result.append(pixel)
                #print result
                writer.writerow(result)