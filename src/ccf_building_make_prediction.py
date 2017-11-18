from __future__ import division

import os
from tqdm import tqdm
import pandas as pd
import extra_functions
import shapely.geometry
from numba import jit
import tifffile as tiff
from scipy import misc
import cv2
from keras import backend as K
from keras.models import model_from_json
import numpy as np
K.set_image_dim_ordering('th')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def scale_percentile(matrix):
    matrix.transpose([1,2,0])
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 0, axis=0)
    maxs = np.percentile(matrix, 100, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix

def read_model(cross=''):
    json_name = 'architecture_128_4_buildings_n3b_v2' + cross + '.json'
    weight_name = 'model_weights_128_4_buildings_n3b_v2' + cross + '.h5'
    model = model_from_json(open(os.path.join('../cache', json_name)).read())
    model.load_weights(os.path.join('../cache', weight_name))
    return model

model = read_model()

sample = pd.read_csv('../data/sample_submission.csv')

data_path = '../data'
num_channels = 3
num_mask_channels = 1
threashold = 0.3

three_band_path = os.path.join(data_path, 'three_band')

train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))
gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))
# modified
#test_ids = ['0_1', '0_4', '1_3']
#test_ids = ['quickbird2015_preliminary_2','quickbird2017_preliminary_2']
test_ids = ['1_8bits','2_8bits','3_8bits']
#test_ids = ['1_8bits']
result = []

def read_image_4(image_id):

   # img_3 = np.transpose(tiff.imread("../data/all/predict/{}.tif".format(image_id)), (1, 2, 0)) / 65535.0
    img_3 = misc.imread("../data/all/new/BDCI2017-jiage/CCF-testing/{}.png".format(image_id)) / 255.0
    result = np.transpose(img_3, (2, 0, 1))

    return result.astype(np.float16)


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


for image_id in tqdm(test_ids):

    image = read_image_4(image_id)

    H = image.shape[1]
    W = image.shape[2]


    predicted_mask = extra_functions.make_prediction_cropped(model, image, initial_size=(112, 112),
                                                             final_size=(112-32, 112-32),
                                                             num_masks=num_mask_channels, num_channels=num_channels)

    image_v = flip_axis(image, 1)
    predicted_mask_v = extra_functions.make_prediction_cropped(model, image_v, initial_size=(112, 112),
                                                               final_size=(112 - 32, 112 - 32),
                                                               num_masks=1,
                                                               num_channels=num_channels)

    image_h = flip_axis(image, 2)
    predicted_mask_h = extra_functions.make_prediction_cropped(model, image_h, initial_size=(112, 112),
                                                               final_size=(112 - 32, 112 - 32),
                                                               num_masks=1,
                                                               num_channels=num_channels)

    image_s = image.swapaxes(1, 2)
    predicted_mask_s = extra_functions.make_prediction_cropped(model, image_s, initial_size=(112, 112),
                                                               final_size=(112 - 32, 112 - 32),
                                                               num_masks=1,
                                                               num_channels=num_channels)

    new_mask = np.power(predicted_mask *
                        flip_axis(predicted_mask_v, 1) *
                        flip_axis(predicted_mask_h, 2) *
                        predicted_mask_s.swapaxes(1, 2), 0.25)
    

    mask_1 = new_mask[0]>threashold
    mask_1 = ((mask_1 == 1) * 255).astype(np.uint8)
    misc.imsave("../data/all/new/BDCI2017-jiage/CCF-testing/{}_building_predict.png".format(image_id), mask_1)




