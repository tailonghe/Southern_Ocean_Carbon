import numpy as np
from tensorflow.keras.optimizers import Adam
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
import argparse
import sys
import glob
from datetime import datetime

from model.core import RUnet_model
from utils.functions import r_score, mse_nonzero, data_split
from utils.data_generator import data_generator



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a SOC model')

    parser.add_argument('--x', dest='x_files',
                        help='Filenames of x data',
                        default=None, type=str, nargs='+')

    parser.add_argument('--y', dest='y_files',
                        help='Filenames of y data',
                        default=None, type=str, nargs='+')

    parser.add_argument('--lvl1', dest='level1',
                        help='First (upper) layer',
                        default=None, type=int, nargs=1)

    parser.add_argument('--lvl2', dest='level2',
                        help='Second (lower) layer',
                        default=None, type=int, nargs=1)


    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")

    parser.add_argument('--o', dest='outname',
                        help='Trained model name',
                        default='DLSOC_model_trained_' + dt_string + '.h5', type=str, nargs='?')

    parser.add_argument('--lr', dest='lr',
                        help='Learning rate',
                        default=1e-5, type=float, nargs='?')

    parser.add_argument('--w', dest='weights',
                        help='Pretrained model weights',
                        default=None, type=str, nargs='?')

    parser.add_argument('--b', dest='batch_size',
                        help='Size for batch training',
                        default=5, type=int, nargs='?')


    if len(sys.argv) < 5:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if (not args.x_files) or (not args.y_files):
        parser.print_help()
        sys.exit(1)

    if (not args.level1) or (not args.level2):
        parser.print_help()
        sys.exit(1)

    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args: ')
    print('X filenames: ', args.x_files)
    print('Y filenames: ', args.y_files)
    print('First level: ', args.level1)
    print('Second level: ', args.level2)
    print('Batch size: ', args.batch_size)
    print('Learning rate: ', args.lr)
    print('Output model name: ', args.outname)

    _xfiles = np.array(sorted(args.x_files))
    _yfiles = np.array(sorted(args.y_files))
    _level1 = args.level1[0]
    _level2 = args.level2[0]
    _batchsize = args.batch_size
    _lr = args.lr

    soc_model = RUnet_model(_level1, _level2)
    opt = Adam(lr=_lr) 
    soc_model.compile(optimizer=opt, loss=mse_nonzero, metrics=[r_score, mse_nonzero])
    soc_model.info()

    # load pretrained model weights if provided
    if args.weights:
        soc_model.load_weights(args.weights)

    xtrain, ytrain, xvalid, yvalid, xtest, ytest = data_split(_xfiles, _yfiles, 0.7225, 0.85, maskname='sample_train_valid_test_mask.npz')
    train_generator = data_generator(xtrain, ytrain, _level1, _level2, batch_size=_batchsize)
    valid_generator = data_generator(xvalid, yvalid, _level1, _level2, batch_size=_batchsize)


    csv_logger = CSVLogger( 'sample_log.csv' , append=True, separator=';')
    earlystopper = EarlyStopping(patience=20, verbose=1)
    checkpointer = ModelCheckpoint('checkpt_{val_loss:.2e}_example.h5', verbose=1, save_best_only=True)
    soc_model.train(train_generator,
                        validation_data=valid_generator, epochs=250,
                        callbacks=[earlystopper, checkpointer, csv_logger])

    soc_model.save_model(args.outname)