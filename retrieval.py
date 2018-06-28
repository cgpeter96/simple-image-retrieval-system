#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle

import numpy as np
from keras import backend as K
from keras.applications import VGG16
from keras.models import Model
from keras.preprocessing import image
from scipy.linalg import norm
from keras.models import load_model
'''
#old code
model = VGG16()
model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
'''
MODEL_PATH = 'models/model-ep339.h5'
model = load_model(MODEL_PATH)

model.layers.pop()
model.layers.pop()
model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
def extract_feature(img_path):
    input_shape = (224, 224, 3)
   
    img = image.load_img(img_path, target_size=input_shape)
    img = np.expand_dims(img, axis=0)
    # img = preprocess_input(img)
    feature = model.predict(img)
    # return feature
    
    # print(feature)
    # print(feature.shape)
    norm_feat = feature[0] / norm(feature[0])
    return norm_feat


def create_datasets(dir):
    img_path = [os.path.join(dir, i) for i in os.listdir(dir)]
    # print(img_path)
    # print(extract_feature(img_path[0]).shape)
    features = np.vstack([extract_feature(i) for i in img_path])
    names = [os.path.split(i)[1] for i in img_path]
    return features, names


def save_features(features):
    # h5f = h5py.File('features_gray','w')
    # h5f.create_dataset('features',features[0])
    # names = np.string_(features[1])
    # h5f.create_dataset('names',names)
    # h5f.close()
    pickle.dump(features[0], open('features.pkl', 'wb'))
    pickle.dump(features[1], open('name.pkl', 'wb'))


# TODO  design data
def triplet_loss(anchor, positive, negative, alpha):
    '''
    Calculate the triplet loss according to the FaceNet paper
    :param anchor: the embedding for the anchor image
    :param positive: the embedding for the positive image
    :param negative: the embedding for the negative image
    :param alpha: the fine-tuning feature
    :return: the triplet loss according to the FaceNet paper as a float tensor
    '''
    # with tf.variable_scope('triplet_loss'):
    pos_dist = K.mean(np.sum(K.square(np.subtract(anchor, positive))), axis=1)
    neg_dist = K.mean(np.sum(K.square(np.subtract(anchor, negative))), axis=1)
    basic_loss = np.add(np.subtract(pos_dist, neg_dist), alpha)
    loss = np.mean(np.sum(np.maximum(basic_loss, 0.0), 0))
    return loss


if __name__ == '__main__':
    features = create_datasets('./clothes/')
    save_features(features)
    print('OK!')
