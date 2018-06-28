#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import retrieval
from sklearn.neighbors import KDTree

names = pickle.load(open('name.pkl', 'rb'))
features = pickle.load(open('features.pkl', 'rb'))
tree = KDTree(features)

def search_images(image_path):

    query_image = retrieval.extract_feature(image_path)
    query_image = np.expand_dims(query_image, 0)
    a=np.linalg.norm(features,axis=1)
    b=np.linalg.norm(query_image,axis=1)
    scores = np.dot(features,query_image.T).flatten()
    scores = 1-(scores.flatten()/(a*b))
    # scores = np.sqrt(np.sum((features - query_image) ** 2, axis=1))
    rank_id = np.argsort(scores)
    top4_id = rank_id[:6]
    fname = np.array(names)[top4_id]
    return fname

def search_images2(image_path):
    

    query_image = retrieval.extract_feature(image_path)
    query_image = np.expand_dims(query_image, axis=0)
    top4_id = tree.query(query_image, k=4, return_distance=False).flatten()

    # TODO cos or edcu
    # scores = np.sqrt(np.sum((features - query_image) ** 2, axis=1))
    # rank_id = np.argsort(scores)
    # top4_id = rank_id[:4]
    fname = np.array(names)[top4_id]
    # database = 'database2/'
    # img_path = [os.path.join(database, i) for i in fname]
    # return img_path
    return fname

if __name__ == '__main__':
    q=search_images(r'D:\py_workspace\image_retrieval\clothes2\id_00000003_02_7_additional.jpg')
    
    print(q)

