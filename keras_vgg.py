from keras.models import *
from keras.layers import *
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras.callbacks import *
from keras import backend as K
from keras import regularizers
from online_data import *
import os
from keras.preprocessing import image

os.environ['CUDA_EDVICE_ORDER'] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = "0" # 1 就是gpu1

def VGG16(include_top=True,
          input_shape=(224, 224, 3),
          pooling=None,
          classes=23):
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # if include_top:
        # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    # else:
    #     if pooling == 'avg':
    #         x = GlobalAveragePooling2D()(x)
    #     elif pooling == 'max':
    #         x = GlobalMaxPooling2D()(x)

    model = Model(inputs=img_input, outputs=x)
    
    return model


class VGG(object):
    def __init__(self, train=True):
        self.num_classes = 23
        self.weight_decay = 0.0005
        self.x_shape = [224, 224, 3]
        self.model = self.build_model()
        if train:
            self.train(self.model)
        else:
            self.model.load_weights('keras_vgg.h5')

    def build_model(self):
        # return VGG16(input_shape=self.x_shape, classes=self.num_classes)
        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model

    def normalize(self, X_train, X_test):
        mean = np.mean(X_train, axis=(0, 1, 2, 3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - mean) / (std + 1e-7)
        X_test = (X_test - mean) / (std + 1e-7)
        return X_train, X_test

    def normalize_production(self, x):
        mean = 120.707
        std = 64.15
        return (x - mean) / (std + 1e-7)

    def predict(self, x, normalize=True, batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x, batch_size)

    def predict_image(self, img_path, normalize=False):
        img = image.load_img(img_path, target_sizes=self.x_shape)
        if normalize:
            img = self.normalize_production(img)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return self.model.predict(img)

    def train(self, model):
        batch_size = 64
        maxepoches = 250
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = LearningRateScheduler(lr_scheduler)

        # 定于数据处理器
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)

       

        

        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # 回调函数
        tb = TensorBoard(log_dir='./logs/', histogram_freq=0, write_graph=True, write_images=True, embeddings_freq=0,
                         embeddings_layer_names=None, embeddings_metadata=None)

        # 给数据
        historytemp = model.fit_generator( datagen.flow_from_directory('image/',target_size=(224,224),batch_size=32,class_mode='categorical'),
                                          steps_per_epoch= 100,
                                          epochs=maxepoches,
                                          callbacks=[reduce_lr], verbose=2)

        model.save_weights('keras_vgg.h5')

        return model


if __name__ == '__main__':
    

    model = VGG(False)

    # predicted_x = model.predict(x_test)
    # residuals = np.argmax(predicted_x, 1) != np.argmax(y_test, 1)

    # loss = sum(residuals) / len(residuals)
    # print("the validation 0/1 loss is: ", loss)

