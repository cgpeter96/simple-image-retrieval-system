import numpy as np
from keras.preprocessing.image import *

from keras.models import load_model
from scipy.misc import *
import numpy as np

label = {
    '男士,牛仔裤 (MEN-Denim)': 0,
    '男士,夹克&马甲 (MEN,Jackets&Vests)': 1,
    '男士,裤子(MEN-Pants)': 2,
    '男士，衬衫&polo衫(MEN-Shirts_Polos)': 3,
    '男士，短裤(MEN-Shorts)': 4,
    '男士,西装(MEN-Suiting)': 5,
    '男士，毛衣(MEN-Sweaters)': 6,
    '男士，运动衫&帽衫(MEN-Sweatshirts_Hoodies)': 7,
    '男士,背心&T恤(MEN,Tees&Tanks)': 8,
    '女士，短衫&衬衫(WOMEN-Blouses_Shirts)': 9,
    '女士，开襟衫(WOMEN-Cardigans)': 10,
    '女士，牛仔裤(WOMEN-Denim)': 11,
    '女士，连衣裙(WOMEN-Dresses)': 12,
    '女士，图案T恤(WOMEN-Graphic_Tees)': 13,
    '女士，夹克&外套(WOMEN-Jackets_Coats)': 14,
    '女士，收腿裤(WOMEN-Leggings)': 15,
    '女士，裤子(WOMEN-Pants)': 16,
    '女士，连身裤&连身衣(WOMEN-Rompers_Jumpsuits)': 17,
    '女士，短裤(WOMEN-Shorts)': 18,
    '女士，短裙(WOMEN-Skirts)': 19,
    '女士，毛衣(WOMEN-Sweaters)': 20,
    '女士，运动衫&帽衫(WOMEN-Sweatshirts_Hoodies)': 21,
    '女士，背心&T恤(WOMEN-Tees_Tanks)': 22}

MODEL_PATH = 'models/model-ep339.h5'
label = dict([(v, k) for k, v in label.items()])
model = load_model(MODEL_PATH)


def deal_image(img_path):
    img = imread(img_path)
    img = imresize(img, (224, 224, 3))
    img = np.expand_dims(img, axis=0)
    return img


def classifier(img_path):
    img = deal_image(img_path)
    preds = model.predict(img)
    result = label[np.argmax(preds)]
    return result
