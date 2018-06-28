#!/usr/bin/env python
# -*- coding: utf-8 -*-

# img = Image.open('1_1.png')
# img = np.asarray(img)
import os
import shutil

import matplotlib.pyplot as plt
from PIL import Image

p = './data/'
t = 'tmp_imgs/'
img_path = [os.path.join(p,i) for i in os.listdir(p)]
if os.path.exists(t):
    shutil.rmtree(t)
    os.mkdir(t)
else:
    os.mkdir(t)

imgs = [Image.open(i).convert('L')for i in img_path]

# imgs = [cv2.cvtColor(cv2.imread(i),cv2.COLOR_BGR2GRAY) for i in img_path]
index = 1

for i in imgs:
    plt.imsave('tmp_imgs/'+str(index)+'.jpg',i)
    index += 1