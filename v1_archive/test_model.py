from scripts.model import SOC_model
import numpy as np
from scripts.utils import r2_score, nonzero_mse, data_split
from keras.optimizers import Adam
import argparse
import sys


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a SOC model')

    parser.add_argument('--x', dest='x_files',
                        help='Filenames of x data',
                        default=None, type=str, nargs='+')
    parser.add_argument('--w', dest='model_name',
                        help='Name of pretrained model weights',
                        default=None, type=str, nargs=1)
    parser.add_argument('--o', dest='outfolder',
                        help='Name of output folder for predictions',
                        default='test_output/', type=str, nargs='?')


    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with args: ')
    print('Predictors: ', args.x_files)
    print('Used model: ', args.model_name)
    print('Output directory: ', args.outfolder)

    soc_model = SOC_model()
    opt = Adam() 
    soc_model.compile(optimizer=opt, loss=nonzero_mse, metrics=[r2_score, nonzero_mse])
    soc_model.summary()

    soc_model.load_weights(args.model_name[0])


    for f in args.x_files:
        X = np.load(f)
        Y = soc_model.predict(X)
        np.save( args.outfolder + 'predictions_for_' + f, Y)











