from __future__ import division

import sys
import numpy as np 
from scipy import misc
from tqdm import tqdm

building_file_name = {'1-8bits':0, '2-8bits':1, 'train1_labels_8bits_building_mask':2, 'train2_labels_8bits_building_mask':3, 'train3_labels_8bits_building_mask':4}
road_file_name = {'1-8bits':0, '2-8bits':1, 'train1_labels_8bits_road_mask':2, 'train2_labels_8bits_road_mask':3, 'train3_labels_8bits_road_mask':4}
water_file_name = {'1-8bits':0, '2-8bits':1, 'train1_labels_8bits_water_mask':2, 'train2_labels_8bits_water_mask':3, 'train3_labels_8bits_water_mask':4}
vegetation_file_name = {'1-8bits':0, '2-8bits':1, 'train1_labels_8bits_vegetation_mask':2, 'train2_labels_8bits_vegetation_mask':3, 'train3_labels_8bits_vegetation_mask':4}

all_file_name = {'mask_building/': building_file_name, 'mask_road/':road_file_name, 'mask_water/': water_file_name, 'mask_vegetation/': vegetation_file_name}

# directory_4_classes = ['mask_building/', 'mask_road/', 'mask_water/', 'mask_vegetation/']

directory_save = {'mask_building/': 'building_mask/', 'mask_road/': 'road_mask/', 'mask_water/': 'water_mask/', 'mask_vegetation/': 'vegetation_mask/'}
save_file_name = ['1-8bits-train', '2-8bits-train', '1_8bits_train_semi', '2_8bits_train_semi', '3_8bits_train_semi']
save_prefix = '../ccf_all/mask_cut/'
save_postfix = '.png'

def parse_agrs():
    if len(sys.argv) == 2:
        stride =int(sys.argv[1])
    else:
        stride = 1000  #default
    return stride

def read_img(file_name, directory):
	img = {}
	prefix = '../ccf_all/mask/'
	postfix = '.png'
	for fn, n in tqdm(file_name.items()):
		tempt_img = misc.imread(prefix+directory+fn+postfix)
		# print tempt_img.shape
		# print fn
		img[fn] = tempt_img

	return img

def cut_and_save_single_img(img, dir_prefix, fn, cut_size=1000):
	row = int(img.shape[0] / cut_size + 1)
	col = int(img.shape[1] / cut_size + 1)

	rows = img.shape[0]
	colms = img.shape[1]
	for i in range(row):
		for j in range(col):
			FileName = fn+str(i)+'_'+str(j)
			if(i != row-1):
				if(j != col-1):
					tempt_img = img[i*cut_size:(i+1)*cut_size, j*cut_size:(j+1)*cut_size]

				else:
					tempt_img = img[i*cut_size:(i+1)*cut_size, colms-cut_size:colms]
					
			else:
				if(j != col-1):
					tempt_img = img[rows-cut_size:rows, j*cut_size:(j+1)*cut_size]

				else:
					tempt_img = img[rows-cut_size:rows, colms-cut_size:colms]
			misc.imsave(save_prefix+dir_prefix+FileName+save_postfix, tempt_img)


def cut_and_save_the_whole_list(img_list, filename, dir_prefix,cut_size=1000):
	
	for fn, il in tqdm(img_list.items()):
		# print il.shape
		cut_and_save_single_img(il, directory_save[dir_prefix], save_file_name[filename[fn]], cut_size)
		

def cut_and_save_the_all(cut_size=1000):

	for dir_, filename in all_file_name.items():
		print 'Now We will cut the ' + dir_[:-1] + '.'
		print 'reading data.....'

		img_list = read_img(filename, dir_)

		print 'Cutting and saving...Please wait patiently...'
		cut_and_save_the_whole_list(img_list, filename, dir_, cut_size)
		print 'Done ' + dir_[:-1] +'.'

if __name__ == '__main__':

	cut_size = parse_agrs()
	
	cut_and_save_the_all(cut_size)

	print 'Have done all! Thank you!'
