import numpy as np
from keras.preprocessing.image import *

from keras.models import load_model
from scipy.misc import *
import os
import pickle
from keras import backend as K
from keras.applications import VGG16
from keras.models import Model
from keras.preprocessing import image
from scipy.linalg import norm

MODEL_PATH = 'models/model-ep339.h5'
model = load_model(MODEL_PATH)

model.layers.pop()
model.layers.pop()
model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
print(model)
def extract_feature(img_path):
    input_shape = (224, 224, 3)
   
    img = image.load_img(img_path, target_size=input_shape)
    img = np.expand_dims(img, axis=0)
    # img = preprocess_input(img)
    feature = model.predict(img)
    # return feature
    norm_feat = feature[0] / norm(feature[0])
    return norm_feat

img_path='G:/BaiduYunDownload/In-shop Clothes Retrieval Benchmark/Img/img/img/WOMEN/Graphic_Tees/id_00000019/01_1_front.jpg'

print(extract_feature(img_path))