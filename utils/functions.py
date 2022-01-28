from keras import backend as K
import tensorflow as tf
import numpy as np
import random

def r_score(y_true, y_pred):
    """ Pearson correlation coefficient, calculated between truth and prediction.
    """
    y_t = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, 0), tf.float32))
    y_p = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, 0), tf.float32))

    SS_res =  K.sum(K.square(y_t - y_p))
    SS_tot = K.sum(K.square(y_t - K.mean(y_t)))

    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def mse_nonzero(y_true, y_pred):
    """ Mean squared error loss without contribution from masked regions.
    """
    y_t = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, 0), tf.float32))
    y_p = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, 0), tf.float32))
    return K.mean(K.square(y_p - y_t), axis=-1)



def process_x(x):
    """ Apply scaling factors to features, and mask bad data with zero
    """
    tempx = x.copy()
    tempx[np.where(np.isnan(tempx))] = 0
    tempx[:, :, :,  2] = tempx[:, :, :,  2]*1e-3  # 
    return tempx

def add_buffer(data, direction=1):
    """ Add buffers to the longitude dimension of the South Ocean domain. +/- 8 longitude deg.
    direction = 1: add buffer
    direction = -1: remove buffer
    """
    # data should be in shape of (batch_size, lat, lon, lev/feat)
    temp = data.copy()
    if direction == 1:
        head = temp[:, :, :8, :]
        tail = temp[:, :, -8:, :]
        temp = np.concatenate([temp, head], axis=2)
        temp = np.concatenate([tail, temp], axis=2)
    else:
        return temp[:, :, 8:-8, :]
    return temp



def data_split(x, y, ratio1, ratio2, maskname=None):
    """ Split data set into train, valid, test sets.
    ratio1: percentage of training set out of whole data set
    ratio2: percentage of training+valid set out of whole data set
    maskname: name of the mask file to be saved
    """

    dsize1 = int(x.shape[0]*ratio1)
    dsize2 = int(x.shape[0]*ratio2)
    dmask = np.array(list(range(0, x.shape[0])))
    random.shuffle(dmask)

    dmask1 = dmask[:dsize1]
    x1 = x[dmask1]
    y1 = y[dmask1]

    dmask2 = dmask[dsize1:dsize2]
    x2 = x[dmask2]
    y2 = y[dmask2]

    dmask3 = dmask[dsize2:]
    x3 = x[dmask3]
    y3 = y[dmask3]

    if maskname:
        np.savez(maskname, train=dmask1, valid=dmask2, test=dmask3)

    return x1, y1, x2, y2, x3, y3
