#coding=utf-8

import cv2
import numpy as np
import random


#先上采用再下采用，用于降低分辨率
def lower_resolution(im,scale=2):
    im = cv2.resize(im,(0,0),fx=scale,fy=scale)
    im = cv2.resize(im,(0,0),fx=1/scale,fy=1/scale)
    return im
    
'''
定义gamma变换函数：
gamma就是Gamma
'''
def gamma_transform(img, gamma):
    img = img.astype(np.uint8)
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

'''
随机gamma变换，调整光照
gamma_vari是Gamma变化的范围[1/gamma_vari, gamma_vari)
'''
def random_gamma_transform(src, gamma_vari=2.0):
    img = src.copy()
    img = img.transpose(1,2,0)
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    #print 'gamma:',img.shape
    img = gamma_transform(img, gamma)
    img = img.transpose(2,0,1)
    #print 'gamma:',img.shape
    img = img.astype(np.float64)
    return img

'''
定义hsv变换函数：
hue_delta是色调变化比例
sat_delta是饱和度变化比例
val_delta是明度变化比例
'''
def hsv_transform(img, hue_delta, sat_mult, val_mult):
    img = img.astype(np.uint8)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] *= sat_mult
    img_hsv[:, :, 2] *= val_mult
    img_hsv[img_hsv > 255] = 255
    return cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)

'''
随机hsv变换,色彩抖动
hue_vari是色调变化比例的范围
sat_vari是饱和度变化比例的范围
val_vari是明度变化比例的范围
'''
def random_hsv_transform(src, hue_vari=10, sat_vari=0.1, val_vari=0.1):
    img = src.copy()
    img = img.transpose(1,2,0)
    hue_delta = np.random.randint(-hue_vari, hue_vari)
    sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
    val_mult = 1 + np.random.uniform(-val_vari, val_vari)
    img = hsv_transform(img, hue_delta, sat_mult, val_mult)
    img = img.transpose(2,0,1)
    img = img.astype(np.float64)
    return img
    
def add_noise(src):
    img = src.copy()
    img = img.transpose(1,2,0)
    for i in range(20): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    img = img.transpose(2,0,1)
    return img

def flip(img,mask):
    var = np.random.random()
    if var < 0.3:
        # 横向翻转图像
        flipped_img = cv2.flip(img, 1)
        flipped_mask = cv2.flip(mask, 1)
    elif var < 0.6:
        # 纵向翻转图像
        flipped_img = cv2.flip(img, 0)
        flipped_mask = cv2.flip(mask, 0)
    else:
        # 同时在横向和纵向翻转图像
        flipped_img = cv2.flip(img, -1)
        flipped_mask = cv2.flip(mask, -1)
    return flipped_img,flipped_mask

def r(factor):
    return int(np.random.random() * factor);


def Addunbalance(img):
    for x in xrange(img.shape[0]):
        img[x, :] -= -10 + r(10)
    return img
    
def blur(src):
    img = src.copy()
    img = img.transpose(1,2,0)
    img = cv2.blur(img, (3, 3));
    img = img.transpose(2,0,1)
    return img

def data_augment(xb):
    # if np.random.random() < 0.5:
        # xb = flip_axis(xb, 1)
        # yb = flip_axis(yb, 1)

    # if np.random.random() < 0.5:
        # xb = flip_axis(xb, 2)
        # yb = flip_axis(yb, 2)

    # if np.random.random() < 0.5:
        # xb = xb.swapaxes(1, 2)
        # yb = yb.swapaxes(1, 2)    
    
    if np.random.random() < 0.5:
        xb = blur(xb)
    
    #if np.random.random() < 0.5:
        #xb = add_noise(xb)
    
    #if np.random.random() < 0.5:
        #xb = Addunbalance(xb)    

    #if np.random.random() < 0.5:
        #xb = random_hsv_transform(xb)     
       
    #if np.random.random() < 0.5:
        #xb = random_gamma_transform(xb)    

    return xb       
        
def test():
    img = cv2.imread('train2_8bits.png')
    mask = cv2.imread('visualized_train2_labels_8bits_building_mask.png')
    
    img_rows = 112
    img_cols = 112
    X_height = img.shape[0]
    X_width = img.shape[1]
    print 'image shape:',img.shape
    
    for i in range(200):   
        random_width = random.randint(0, X_width - img_cols - 1)
        random_height = random.randint(0, X_height - img_rows - 1)
        img_roi = np.array(img[random_height: random_height + img_rows, random_width: random_width + img_cols,:])
        mask_roi = mask[random_height: random_height + img_rows, random_width: random_width + img_cols,:]
        dst = data_augment(img_roi) 
        cv2.imwrite('./test6/'+str(i)+'.png',dst)
        cv2.imwrite('./test6/mask_'+str(i)+'.png',mask_roi)


#test() 
