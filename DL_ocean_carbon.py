""" Code for the Southern Ocean Carbon project
"""
__author__ = "Tai-Long He, U of Toronto"
__email__ = "tailong.he@mail.utoronto.ca"
__homepage__="https://tailonghe.github.io/"

import glob
import numpy as np
from numpy.random import seed, randint
import tensorflow as tf
from keras.utils import Sequence
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense
from keras import backend as K


class data_generator( Sequence ) :
    """
    Daga generator for phase 1 training using B-SOSE simulation data, used to save computational resources.
    """
  
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
    return tempx, tempy



def r2_keras(y_true, y_pred):
    """
    Function to calculate r2 score between prediction and truth
    INPUTS:
        y_true: array of truth
        y_pred: array of predictions
    OUTPUT:
        r2: computed r2 score
    """

    # filter out missing values
    y_t = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, 0), tf.float32))
    y_p = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, 0), tf.float32))

    SS_res =  K.sum(K.square(y_t - y_p)) 
    SS_tot = K.sum(K.square(y_t - K.mean(y_t))) 
    r2 = ( 1 - SS_res/(SS_tot + K.epsilon()) )

    return r2
  

def nonzero_MSE(y_true, y_pred):
    """
    Function to calculate mean squared loss for non-zero values in the output
    INPUTS:
        y_true: array of truth
        y_pred: array of predictions
    OUTPUT:
        mse: computed mean squared errors
    """

    # filter out missing values
    y_t = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, 0), tf.float32))
    y_p = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, 0), tf.float32))
    mse = K.sum(K.square(y_p - y_t), axis=-1)
    return mse


def fname1(fname):
    """
    Function to return input file name for SOCCOM data set
    """
    seq1 = fname.split('/')
    yy = seq1[1][:4]
    yname = 'SOCCOM_Y_clean/' +yy+ '_smoothed.npy' 
    return yname

def fname2(fname):
    """
    Function to return input file name for Argo data set
    """
    seq1 = fname.split('/')
    yy = seq1[1][:4]
    yname = 'Argo_Y_clean/' +yy+ '_all_smoothed.npy' 
    return yname


def outname1(fname):
    """
    Function to return input file name for SOCCOM predictions
    """
    seq1 = fname.split('/')
    yy = seq1[1][:4]
    yname = 'bottle_Y_pred_4km_both/' + yy + '_pred_both.npy'
    return yname


def outname2(fname):
    """
    Function to return input file name for Argo predictions
    """
    seq1 = fname.split('/')
    yy = seq1[1][:4]
    yname = 'Argo_Y_new_pred_4km/' + yy + '_pred_both.npy'
    return yname


# Prepare X-Y data pairs
xfiles1 = sorted(glob.glob('SOCCOM_X_clean/*.npy'))
xfiles2 = sorted(glob.glob('Argo_X_clean/*.npy'))

xfiles = xfiles1 + xfiles2
yfiles = [fname1(s) for s in xfiles1] + [fname2(s) for s in xfiles2]

bottlex = np.concatenate([np.load(f) for f in xfiles1], axis=0)
bottlesize = bottlex.shape[0]
argox = np.concatenate([np.load(f) for f in xfiles2], axis=0)
argosize = argox.shape[0]
print('SOCCOM data set size: ', bottlesize)
print('Argo data set size: ', argosize)

X = np.concatenate([np.load(f) for f in xfiles], axis=0)
Y = np.concatenate([np.load(f)[:, :, :48] for f in yfiles], axis= 0)

# scale the pCO2 variable to match the unit in B-SOSE 
X[:, :, 1] = X[:, :, 1] * 1e-6
# mark missing values as zeros (filtered out in training)
Y = Y[:, :, :48] 
Y[np.where(Y < 0)] = 0
Y[np.where(np.isnan(Y))] = 0


print(X.shape, Y.shape)
print('Y False check: ', np.any(np.isnan(Y)))
print('X False check: ', np.any(np.isnan(X)))


# Build the DL model
inputs = Input((1, 10), name='inputs')
d1 = Dense(128, activation="linear")(inputs) 
c1 = Conv1D(128, 2, strides=1, padding='same', activation="relu")(d1)
c2 = Conv1D(128, 2, strides=1, padding='same', activation="relu")(c1)

d2 = Dense(256, activation="linear")(c2) 
c3 = Conv1D(256, 2, strides=1, padding='same', activation="relu")(d2)
c4 = Conv1D(256, 2, strides=1, padding='same', activation="relu")(c3)

d3 = Dense(512, activation="linear")(c4) 
c5 = Conv1D(512, 2, strides=1, padding='same', activation="relu")(d3)
c6 = Conv1D(512, 2, strides=1, padding='same', activation="relu")(c5)

lstm = LSTM(512, return_sequences=True, name='LSTM1') (c6)

d4 = Dense(512, activation="linear")( lstm ) 
u0 = concatenate([c6, d4])
c7 = Conv1D(512, 2, strides=1, padding='same', activation="relu")(u0)
c8 = Conv1D(512, 2, strides=1, padding='same', activation="relu")(c7)

d5 = Dense(512, activation="linear")(c8) 
u1 =  concatenate([c4, d5])
c9 = Conv1D(512, 2, strides=1, padding='same', activation="relu")(u1)
c10 = Conv1D(512, 2, strides=1, padding='same', activation="relu")(c9)

d6 = Dense(256, activation="linear")(c10) 
u2 = concatenate([c2, d6])
c11 = Conv1D(256, 2, strides=1, padding='same', activation="relu")(u2)
c12 = Conv1D(256, 2, strides=1, padding='same', activation="relu")(c11)

d7 = Dense(256, activation="linear")(c12) # 1024
outputs = Dense(48, activation="linear")(d7) # 1024
carbon_model = Model(inputs=inputs, outputs=outputs)

# Compile the model
opt = Adam(lr=1e-5)
carbon_model.compile(optimizer=opt, loss=mymse, metrics=[r2_keras, mymse, 'mape'])
carbon_model.summary()

# Logger, early stopper and checkpointers
csv_logger = CSVLogger('training_log.csv', append=True, separator=';')
earlystopper = EarlyStopping(patience=10, verbose=1)
checkpointer = ModelCheckpoint('checkpoints/checkpt-{epoch:02d}_{val_r2_keras:2.3f}_{val_loss:2.3f}_4km_phase2_both.h5', verbose=1, save_best_only=True)

# Randomly split the data into training, validation and test data sets
trainsize = int(X.shape[0]*0.8)
testsize = int(X.shape[0]*0.2)
validsize = int(X.shape[0]*0.08)
validmask = np.array([False]*X.shape[0])
trainmask = np.array([True]*X.shape[0])

randind=[]
for i in range(testsize):
    r=randint(0, X.shape[0])
    while (r in randind): 
        r=randint(0, X.shape[0])
    randind.append(r)
randind = np.array(randind)
trainmask[randind] = False

randind=[]
for i in range(validsize):
    r=randint(0, X.shape[0])
    while (r in randind) or ( ~trainmask[r]): # validation data is small subset of training data
        r=randint(0, X.shape[0])
    randind.append(r)
randind = np.array(randind)
validmask[randind] = True

testmask = ~trainmask
testmask.dump('test_mask.npy')

# Split data sets
xtrain = X[np.where(  trainmask )]
ytrain = Y[np.where(  trainmask )]
xvalid = X[np.where(  validmask )]
yvalid = Y[np.where(  validmask )]

print("Train size: ", np.sum(trainmask) )
print("Valid size: ", np.sum(validmask) )
print("Test size: ", np.sum(testmask) )

# Train the model
results = final_mod.fit( xtrain, ytrain, validation_data=(xvalid, yvalid), batch_size=15, shuffle=True, epochs=250, callbacks=[earlystopper, checkpointer, csv_logger])

# Save the trained model
final_mod.save('DL_Ocean_Carbon_model.h5')


# Loop over the SOCCOM bottle predictors to generate predicted carbon profiles
for s in xfiles1:
    print('X now: ', s)
    xnow = np.load(s)
    xnow[:, :, 1] = xnow[:, :, 1] * 1e-6
    ynow = final_mod.predict(xnow)
    print('saving now: ', outname1(s), ynow.shape)
    np.save(outname1(s), ynow)

# Loop over the Argo float predictors to generate predicted carbon profiles
for s in xfiles2:
    print('X now: ', s)
    xnow = np.load(s)
    xnow[:, :, 1] = xnow[:, :, 1] * 1e-6
    ynow = final_mod.predict(xnow)
    print('saving now: ', outname2(s), ynow.shape)
    np.save(outname2(s), ynow)
