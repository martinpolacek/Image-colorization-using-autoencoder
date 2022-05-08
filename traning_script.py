# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:59:27 2020

@author: Martin Poláček
"""

"""Veškeré importy"""
import os
import psutil
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Conv2DTranspose, Dense, Dropout
from  tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from skimage.color import rgb2lab
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib
import math
import gc


os.environ['TF_CPP_MIN_LOG_LEVEL']='1' 
os.environ["CUDA_VISIBLE_DEVICES"]="6,7,8,9" #Select GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' 



mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"])
"""Načtení dat"""

#Loading dataset
L = np.load('L_data.npy')
ab = np.load('ab_data.npy')

BATCH_SIZE = 128
VALIDATION_SPLIT = 0.05

size_of_dataset = math.floor(((1 - VALIDATION_SPLIT) * L.shape[0]) / BATCH_SIZE)
size_of_validation = math.floor((L.shape[0] / BATCH_SIZE) - size_of_dataset)

L_train = L[:(size_of_dataset*BATCH_SIZE + 1)]
ab_train = ab[:(size_of_dataset*BATCH_SIZE + 1)]

L_val = L[-(size_of_validation*BATCH_SIZE + 1):]
ab_val = ab[-(size_of_validation*BATCH_SIZE + 1):]


def batch_generator(array_size, batch_size = 96, is_train=False):
    
    global L_train
    global ab_train
    global L_val
    global ab_val
    
    while True:
            # it might be a good idea to shuffle your data before each epoch
            if is_train:
                p = np.random.permutation(L_train.shape[0])
                L_train = L_train[p]
                ab_train = ab_train[p]
                X = L_train
                Y = ab_train
            else:
                X = L_val
                Y = ab_val
                
            for i in range(array_size):
                start = i*batch_size
                end = start + batch_size + 1
                yield X[start:end], Y[start:end]
                


with mirrored_strategy.scope():
    model = Sequential()
    """Encoder"""
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(1024, (3,3), activation='relu', padding='same', strides=2))
    model.add(Conv2DTranspose(512, (3, 3), activation='relu', padding='same', strides=2))
    """Decoder"""
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(256, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))

    model.add(Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    model.compile(optimizer='adamax', loss='mae' , metrics=['accuracy'])
    
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
cp_callback =  tf.keras.callbacks.ModelCheckpoint('model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')

batch_gen = batch_generator(size_of_dataset, BATCH_SIZE, True)
val_batch_gen = batch_generator(size_of_validation - 1, BATCH_SIZE, False)

history = model.fit(batch_gen ,epochs=40, batch_size=BATCH_SIZE,
                    callbacks=[es_callback, cp_callback], steps_per_epoch=size_of_dataset,
                    validation_data=val_batch_gen, validation_steps = size_of_validation, shuffle=True, verbose=2)

print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('model_result.png')

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('model_result_loss.png')