from __future__ import division

import os

from tqdm import tqdm
import h5py
import numpy as np
import tifffile as tiff
from scipy import misc


data_path = '../data'
datasheet = ['1-8bits', '2-8bits'] # fill the name of data!!!!!!

def read_image_3(image_id):

    img = misc.imread("../data/train/{}.png".format(image_id)) / 255.0
    
    result = np.transpose(img, (2, 0, 1))
    return result.astype(np.float16)

def read_mask(image_id):

    mask = misc.imread("../data/mask_road/{}.png".format(image_id))
    mask = mask[np.newaxis,:,:]
    return mask


def cache_train_3():


    num_train = 2 #how many training pictures do this have? the number of data or Just fill: len(datasheet)

    image_rows = 7939 # all picture's rows and cols, just corresponse to xxxx.shape, the best situation is that all the pictures have the same shape.
    image_cols = 7969

    num_channels = 3

    num_mask_channels = 1

    f = h5py.File(os.path.join(data_path, 'chen_train_road.h5'), 'w', compression='blosc:lz4', compression_opts=9)

    imgs = f.create_dataset('train', (num_train, num_channels, image_rows, image_cols), dtype=np.float16)
    imgs_mask = f.create_dataset('train_mask', (num_train, num_mask_channels, image_rows, image_cols), dtype=np.uint8)

    ids = []

    i = 0
    for image_id in tqdm(datasheet):
        image = read_image_3(image_id)
        print(image_id)

        imgs[i] = image[:, :, :]
        imgs_mask[i] = read_mask(image_id)[:, :, :]

        ids += [image_id]
        i += 1

    # fix from there: https://github.com/h5py/h5py/issues/441
    f['train_ids'] = np.array(ids).astype('|S9')

    f.close()


if __name__ == '__main__':
    cache_train_3()
