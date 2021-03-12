from scripts.model import SOC_model
import numpy as np
from scripts.utils import r2_score, nonzero_mse, data_split
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
import argparse
import sys
import glob
from datetime import datetime


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


    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if (not args.x_files) or (not args.y_files):
        parser.print_help()
        sys.exit(1)

    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args: ')
    print('X filenames: ', args.x_files)
    print('Y filenames: ', args.y_files)
    print('Batch size: ', args.batch_size)
    print('Learning rate: ', args.lr)
    print('Output model name: ', args.outname)

    _xfiles = args.x_files
    _yfiles = args.y_files
    _batchsize = args.batch_size
    _lr = args.lr

    soc_model = SOC_model()
    opt = Adam(lr=_lr) 
    soc_model.compile(optimizer=opt, loss=nonzero_mse, metrics=[r2_score, nonzero_mse])
    soc_model.summary()

    if args.weights:
        soc_model.load_weights(args.weights)

    X = np.concatenate([np.load(f) for f in _xfiles], axis=0)
    Y = np.concatenate([np.load(f) for f in _yfiles], axis= 0)
    X[:, :, 1] = X[:, :, 1] * 1e-6
    Y = Y[:, :, :48]
    Y[np.where( np.isnan(Y) )] = 0
    Y[np.where( Y <= 0 )] = 0

    xvalid, yvalid = data_split(X, Y, 0.1)

    csv_logger = CSVLogger('train_DLSOC.csv', append=True, separator=';')
    earlystopper = EarlyStopping(patience=10, verbose=1)
    checkpointer = ModelCheckpoint('SOC_checkpt.h5', verbose=1, save_best_only=True)
    info = soc_model.train(X, Y, validation_data=(xvalid, yvalid), batch_size=_batchsize, epochs=250, shuffle=True, 
                        callbacks=[earlystopper, checkpointer, csv_logger])

    soc_model.save_model(args.outname)

