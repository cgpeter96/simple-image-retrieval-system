#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from keras.applications import VGG16,resnet50
from keras.models import Model
from keras.preprocessing import image
from scipy.linalg import norm


def extract_feature(img_path):
    input_shape = (224,224,3)
    model = VGG16()
    model.layers.pop()
    print(model.inputs)
    model = Model(inputs=model.inputs,outputs=model.layers[-1].output)
    img = image.load_img(img_path,target_size=(224,224,3))
    img = np.expand_dims(img,axis=0)
    # img = preprocess_input(img)
    feature =model.predict(img)
    print(feature.shape)
    norm_feat = feature[0] / norm(feature[0])
    # return feature
    return  norm_feat

if __name__ == '__main__':
    # extract_feature('4.jpg')
    model=resnet50.ResNet50()