#coding=utf-8

"""
准备训练数据
"""

from __future__ import division

import os
import pandas as pd
import extra_functions
from tqdm import tqdm
import h5py
import numpy as np
import tifffile as tiff
from scipy import misc

data_path = '../data'
src_img = ['1-8bits','2-8bits']
mask_img = ['1_class_8bits','2_class_8bits']

mask1_pool = ['1_class_8bits_vegetation_mask',
              '1_class_8bits_road_mask',
              '1_class_8bits_building_mask',
              '1_class_8bits_water_mask']

mask2_pool = ['2_class_8bits_vegetation_mask',
              '2_class_8bits_road_mask',
              '2_class_8bits_building_mask',
              '2_class_8bits_water_mask']



def read_image_3(image_name):

    img = tiff.imread("../data/all/train/{}.tif".format(image_name)) / 255.0
    print 'image shape:',img.shape
   # result = img
    result = np.transpose(img, (2, 0, 1))
    print 'after image change:',result.shape
    res = np.zeros((1,3,7939,7969))
    
    res[0,:,:,:] = res
    return res.astype(np.uint8)

def read_mask(image_name):

    mask = misc.imread("../data/all/mask/{}.tif".format(image_name))
    print 'mask shape:',mask.shape
    mask = mask[np.newaxis,:,:]
    return mask

def generate_mask(mask_pool=[]):
    mask = np.zeros((4,7939,7969))
    
    for mask_channel in range(len(mask_pool)):
        mask[mask_channel,:,:] = read_mask(mask_pool[mask_channel])
    return mask.astype(np.uint8)

    
    
def cache_train_3():

    image_rows = 7939
    image_cols = 7969

    num_channels = 3
    num_train = len(src_img)
    num_mask_channels = 4  #4个ccf mask: vegetation,road,building,water

    f = h5py.File(os.path.join(data_path, 'ccf_train.h5'), 'w', compression='blosc:lz4', compression_opts=9)

    imgs = f.create_dataset('train', (num_train,num_channels, image_rows, image_cols), dtype=np.uint8)  #训练图3通道，8 bits
    imgs_mask = f.create_dataset('train_mask', (num_train,num_mask_channels, image_rows, image_cols), dtype=np.uint8)

    ids = []

    i = 0
    for image_id in tqdm(range(len(src_img))):
        if image_id == 0:
            pool = mask1_pool
        else:
            pool = mask2_pool

        image = read_image_3(src_img[image_id])
        mask = generate_mask(pool)
        imgs[i] = image[:,:image_rows,:image_cols]
        imgs_mask[i] = mask[:,:image_rows,:image_cols]

        ids += [image_id]
        i += 1

    # fix from there: https://github.com/h5py/h5py/issues/441
    f['train_ids'] = np.array(ids).astype('|S9')

    f.close()


if __name__ == '__main__':
    cache_train_3()
