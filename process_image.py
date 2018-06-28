#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil

import cv2

dir='clothes/'
t_dir = 'tmp_imgs2/'
if os.path.exists(t_dir):
    shutil.rmtree(t_dir)
    os.mkdir(t_dir)
else:
    os.mkdir(t_dir)

img_path = [os.path.join(dir,i) for i in os.listdir(dir)]
index=1
for path in img_path:
    image =cv2.imread(path)
    name = os.path.split(path)[-1]
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(t_dir+name.format(index),gray)
    index+=1
print('Done!')
# image = cv2.imread(r'D:\py_workspace\flask-keras-cnn-image-retrieval\database2\01_3_back.jpg')
# cv2.imshow("Original", image)
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("1_blur_by_Median.jpg", gray)
# cv2.imshow("Gray", gray)
# if(cv2.waitKey(0)==27):
#     cv2.destroyAllWindows()