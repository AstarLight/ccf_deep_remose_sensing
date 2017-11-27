from __future__ import division

import sys
import numpy as np 
from scipy import misc
from tqdm import tqdm

file_name = ['1_8bits_train_semi', '2_8bits_train_semi', '3_8bits_train_semi', '1-8bits-train', '2-8bits-train']
save_prefix = '../ccf_all/after_cut/'
save_postfix = '.png'

def parse_agrs():
    if len(sys.argv) == 2:
        stride =int(sys.argv[1])
    else:
        stride = 1000  #default
    return stride

def read_img(file_name):
	img = {}
	prefix = '../ccf_all/train/'
	postfix = '.png'
	for fn in tqdm(file_name):
		tempt_img = misc.imread(prefix+fn+postfix)
		# print tempt_img.shape
		img[fn] = tempt_img

	return img

def cut_and_save_single_img(img, fn, cut_size=1000):
	row = int(img.shape[0] / cut_size + 1)
	col = int(img.shape[1] / cut_size + 1)
	rows = img.shape[0]
	colms = img.shape[1]
	for i in range(row):
		for j in range(col):
			FileName = fn+str(i)+'_'+str(j)
			if(i != row-1):
				if(j != col-1):
					tempt_img = img[i*cut_size:(i+1)*cut_size, j*cut_size:(j+1)*cut_size,:]

				else:
					tempt_img = img[i*cut_size:(i+1)*cut_size, colms-cut_size:colms,:]
					
			else:
				if(j != col-1):
					tempt_img = img[rows-cut_size:rows, j*cut_size:(j+1)*cut_size,:]

				else:
					tempt_img = img[rows-cut_size:rows, colms-cut_size:colms,:]
			misc.imsave(save_prefix+FileName+save_postfix, tempt_img)



def cut_and_save_the_whole_list(img_list, cut_size=1000):

	for fn, il in tqdm(img_list.items()):
		cut_and_save_single_img(il, fn, cut_size)


if __name__ == '__main__':

	cut_size = parse_agrs()

	img_list = read_img(file_name)
	
	
	cut_and_save_the_whole_list(img_list, cut_size)
