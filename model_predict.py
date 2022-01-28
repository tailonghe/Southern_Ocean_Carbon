import numpy as np
from tensorflow.keras.optimizers import Adam
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
import argparse
import sys
import glob

from model.core import RUnet_model
from utils.functions import r_score, mse_nonzero, data_split, process_x, add_buffer
from utils.data_generator import data_generator


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a SOC model')

    parser.add_argument('--x', dest='x_files',
                        help='Filenames of x data',
                        default=None, type=str, nargs='+')
    parser.add_argument('--lvl1', dest='level1',
                        help='First (upper) layer',
                        default=None, type=int, nargs='+')
    parser.add_argument('--lvl2', dest='level2',
                        help='Second (lower) layer',
                        default=None, type=int, nargs='+')
    parser.add_argument('--w', dest='model_name',
                        help='Name of pretrained model weights',
                        default=None, type=str, nargs=1)
    parser.add_argument('--o', dest='outfolder',
                        help='Name of output folder for predictions',
                        default='test_output/', type=str, nargs='?')


    if len(sys.argv) < 5:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    if (not args.x_files) or (not args.level1) or (not args.level2) or (not args.model_name):
        parser.print_help()
        sys.exit(1)

    
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with args: ')
    print('Predictors: ', args.x_files)
    print('Level 1: ', args.level1)
    print('Level 2: ', args.level2)
    print('Model weights: ', args.model_name)
    print('Output directory: ', args.outfolder)

    _level1 = args.level1[0]
    _level2 = args.level2[0]

    soc_model = RUnet_model(_level1, _level2)
    opt = Adam() 
    soc_model.compile(optimizer=opt, loss=mse_nonzero, metrics=[r_score, mse_nonzero])
    soc_model.info()

    soc_model.load_weights(args.model_name[0])


    for x in args.x_files:
        xdata = np.load(x)['x']
        xdata = xdata[np.newaxis, :, :, :]  # fill the batch dimension, 1, 10, 56, 360
        xdata = np.moveaxis(xdata, 1, -1)   # rotate the dimension of the input vector to match the model: 1, 56, 360, 10
        xdata = process_x(xdata)            # apply scaling factors
        xdata = add_buffer(xdata)           # add buffer domain
        pred = soc_model.predict(xdata)           
        pred = add_buffer(pred, direction=-1)[0]   # remove buffer domain
        filename = args.outfolder + '/prediction_' + x.split('_')[-1]
        print(" >>>>>>>>> Saving: ", filename)
        np.save(filename, pred)