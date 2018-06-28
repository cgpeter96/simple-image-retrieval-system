#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import retrieval
# from classify import recongnition
from sklearn.neighbors import KDTree

names = pickle.load(open('name.pkl','rb'))
features = pickle.load(open('features.pkl','rb'))
database = './clothes/'
print(names)
print(features.shape)
query_image_path = './clothes/3.jpg'
# recongnition(database+os.path.split(query_image_path)[-1])
tree = KDTree(features)
query_image = retrieval.extract_feature(query_image_path)
query_image = np.expand_dims(query_image, axis=0)
result = tree.query(query_image, k=4, return_distance=False).flatten()
'''
print(query_image.shape)
# query_image = np.expand_dims(query_image,axis=0)
# TODO
# scores = np.sqrt(np.sum((features-query_image)**2,axis=1))
scores = np.dot(query_image,features.T)
print(scores.shape)

rank_id = np.argsort(scores.flatten())
top4_id = rank_id[:4]
'''

top4_id = result
print(top4_id)
fname = np.array(names)[top4_id]
print(fname)
img_path=[os.path.join(database,i) for i in fname]
print(img_path)
# print(imgs)




# img_path = search_image('4.jpg')
index=1
for i in img_path:
    im = Image.open(i)
    plt.subplot(2,2,(index))
    plt.imshow(im)
    plt.axis('off')
    index+=1

plt.show()
# a=plt.imread(img_path[0])
# plt.imshow(a)
# plt.show()


