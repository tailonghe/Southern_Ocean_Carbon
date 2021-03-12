import os
import sys
import random
import warnings
import glob
#import netCDF4 as nc

import numpy as np
import pandas as pd

from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model, load_model, Sequential
from keras.layers import Input, LSTM, Dense, Permute, Reshape, LeakyReLU, Cropping2D, ZeroPadding2D
from keras.layers import Cropping3D, ZeroPadding3D
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv1D, Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose
from keras.layers.pooling import MaxPooling2D, MaxPooling3D
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import normalize
from keras.regularizers import l2
import keras.layers as kl
from keras.utils import Sequence

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.metrics import mean_squared_error
import tensorflow as tf
from keras.callbacks import CSVLogger
from keras.utils.vis_utils import plot_model

from numpy.random import seed, randint
from sklearn.model_selection import train_test_split



def r2_keras(y_true, y_pred):
    y_t = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, 0), tf.float32))
    y_p = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, 0), tf.float32))

    SS_res =  K.sum(K.square(y_t - y_p)) 
    SS_tot = K.sum(K.square(y_t - K.mean(y_t))) 

    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
  

def mymse(y_true, y_pred):
    y_t = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, 0), tf.float32))
    y_p = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, 0), tf.float32))
  
    return K.sum(K.square(y_p - y_t), axis=-1)



print('Build model now...')

inputs = Input((1, 10), name='inputs')
d1 = Dense(128, activation="linear")(inputs) # 1024
c1 = Conv1D(128, 2, strides=1, padding='same', activation="relu")(d1)
c2 = Conv1D(128, 2, strides=1, padding='same', activation="relu")(c1)

d2 = Dense(256, activation="linear")(c2) # 1024
c3 = Conv1D(256, 2, strides=1, padding='same', activation="relu")(d2)
c4 = Conv1D(256, 2, strides=1, padding='same', activation="relu")(c3)

d3 = Dense(512, activation="linear")(c4) # 1024
c5 = Conv1D(512, 2, strides=1, padding='same', activation="relu")(d3)
c6 = Conv1D(512, 2, strides=1, padding='same', activation="relu")(c5)


lstm = LSTM(512, return_sequences=True, name='LSTM1') (c6)
#lstm = LSTM(512, return_sequences=True, name='LSTM2') (lstm)

d4 = Dense(512, activation="linear")( lstm ) # 1024
u0 = concatenate([c6, d4])
c7 = Conv1D(512, 2, strides=1, padding='same', activation="relu")(u0)
c8 = Conv1D(512, 2, strides=1, padding='same', activation="relu")(c7)

d5 = Dense(512, activation="linear")(c8) # 1024
u1 =  concatenate([c4, d5])
c9 = Conv1D(512, 2, strides=1, padding='same', activation="relu")(u1)
c10 = Conv1D(512, 2, strides=1, padding='same', activation="relu")(c9)

d6 = Dense(256, activation="linear")(c10) # 1024
u2 = concatenate([c2, d6])
c11 = Conv1D(256, 2, strides=1, padding='same', activation="relu")(u2)
c12 = Conv1D(256, 2, strides=1, padding='same', activation="relu")(c11)

d7 = Dense(256, activation="linear")(c12) # 1024
outputs = Dense(48, activation="linear")(d7) # 1024

final_mod = Model(inputs=inputs, outputs=outputs)

opt = Adam(lr=5e-5)
final_mod.compile(optimizer=opt, loss=mymse, metrics=[r2_keras, mymse, 'mape'])
final_mod.summary()

final_mod.load_weights('DLSOCO2_v9_final_4km_phase2_GLODAP+Argo.h5')


xfiles = glob.glob('1998_2019_predictors/X_predictors_*.npy')

print(xfiles)

for f in xfiles:
    X = np.load(f)
    X[:, :, :, 1] = X[:, :, :, 1] * 1e-6
    print(X.shape) # (~~~, 56, 360, 10)
    Y = np.zeros((X.shape[0], 56, 360, 48))

    xprev = None
    for i in range(73):
        print(i)

        xnow = X[i] # (56, 360, 10)
        print( np.all(xnow == xprev) )
        xprev = xnow

        mask = ~np.isnan(xnow)  # 10 --> all non-nan
        mask = np.sum(mask, axis=-1)
    
        xnow = xnow.reshape((-1, 1, 10))

        ynow = final_mod.predict(xnow)

        ynow = ynow.reshape((56, 360, 48))

        ynow[ np.where(mask < 10) ] = np.nan
        Y[i] = ynow

    np.save('DIC_' + f[13:-4] + '.npy', Y)
