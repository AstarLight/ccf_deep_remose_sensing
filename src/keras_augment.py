#-*- coding: utf-8 -*-

import cv2
from PIL import Image
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def data_augment(im,mask):   
    print 'augment!'
    datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    x = im
    y = mask
    x = x.reshape((1,) + x.shape)  
    y = x.reshape((1,) + y.shape)  
    print 'reshape x:',x.shape
    print 'reshape y:',y.shape

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory

    i = 0
    for batch in datagen.flow(x,y,batch_size=1):
        img = batch[0,:,:,:]
        #print '1:',img.shape
        #img.transpose(2,0,1)
        print '2:',img.shape
        dst = array_to_img(img)
        #plt.imsave('./test2/'+str(m)+'.png',dst)
        #cv2.imwrite('./test2/'+str(m)+'.png',dst)
        dst.save('./test3/'+str(m)+'.png')
        i += 1
        if i == 1: #这个20指出要扩增多少个数据
            break  # otherwise the generator would loop indefinitely
    return img


def test()
    img = load_img('train2_8bits.png')
    mask = load_img('visualized_train1_labels_8bits_water_mask.png')
    
    img_rows = 112
    img_cols = 112
    X_height = img.shape[0]
    X_width = img.shape[1]
    print 'image shape:',img.shape
    random_width = random.randint(0, X_width - img_cols - 1)
    random_height = random.randint(0, X_height - img_rows - 1)
    img_roi = np.array(img[random_height: random_height + img_rows, random_width: random_width + img_cols,:])
    mask_roi = mask[random_height: random_height + img_rows, random_width: random_width + img_cols,:]
    data_augment(img_roi,mask_roi)


