import tensorflow as tf
from keras import backend as K
from keras.utils import Sequence
import numpy as np
import random

def r2_score(y_true, y_pred):
    y_t = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, 0), tf.float32))
    y_p = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, 0), tf.float32))

    SS_res =  K.sum(K.square(y_t - y_p)) 
    SS_tot = K.sum(K.square(y_t - K.mean(y_t))) 

    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
  

def nonzero_mse(y_true, y_pred):
    y_t = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, 0), tf.float32))
    y_p = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, 0), tf.float32))
  
    return K.sum(K.square(y_p - y_t), axis=-1)


def data_split(x, y, ratio, maskname=None):
    dsize = int(x.shape[0]*ratio)
    dmask = np.array(list(range(0, x.shape[0])))
    random.shuffle(dmask)
    dmask = dmask[:dsize]
    x = x[dmask]
    y = y[dmask]
    if maskname:
        np.save(savemask, dmask)

    return x, y

class data_generator( Sequence ) :
  
  def __init__(self, xnames, ynames, batch_size) :
    self.xnames = xnames
    self.ynames = ynames
    self.batch_size = batch_size
    self.fidx = 0
    self.fnum = int(batch_size / 317520) + 1
    self.start = 0
    
  def __len__(self) :
    return (np.ceil(len(self.xnames) * 317520 / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    self.fidx = int(idx * self.batch_size / 317520)

    batch_x = self.xnames[ self.fidx : self.fidx + self.fnum ]
    batch_y = self.ynames[ self.fidx : self.fidx + self.fnum ]
    
    tempx = np.concatenate([ np.load(s) for s in batch_x], axis=0)
    tempy = np.concatenate([ np.load(s) for s in batch_y], axis=0)

    tempx = tempx[ self.start : self.start + self.batch_size]
    tempy = tempy[ self.start : self.start + self.batch_size]

    self.start = int(idx * self.batch_size % 317520)

    tempx = np.concatenate( (tempx[:, :, :7], tempx[:, :, -3:]), axis=-1 )
    tempy = tempy[:, :, :48]

    return tempx, tempy