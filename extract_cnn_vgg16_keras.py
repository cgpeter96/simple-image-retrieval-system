# -*- coding: utf-8 -*-
# Author: yongyuan.name

import numpy as np
from numpy import linalg as LA

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

'''
 Use vgg16 model to extract features
 Output normalized feature vector
'''
def extract_feat(img_path):
    # weights: 'imagenet'
    # pooling: 'max' or 'avg'
    # input_shape: (width, height, 3), width and height should >= 48
    
    input_shape = (224, 224, 3)
    model = VGG16(weights = 'imagenet',
                  input_shape = (input_shape[0], input_shape[1], input_shape[2]),
                  pooling = 'max',
                  include_top = False)
        
    img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))

    img = image.img_to_array(img)
    # img = rgb2gray(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feat = model.predict(img)
    # print(feat.shape)
    norm_feat = feat[0]/LA.norm(feat[0])
    return norm_feat


if __name__ == '__main__':
    fp ='1.png'
    img = extract_feat(fp)
    import matplotlib.pyplot as plt
    print(img.shape)
    # plt.imshow(img)
    # plt.show()