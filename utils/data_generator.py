from tensorflow.keras.utils import Sequence
import numpy as np
from utils.functions import process_x, add_buffer

# 0 ~ land/masked, 1 ~ valid area
depth = np.load('utils/3d_tropo_mask.npz')['mask']  # 56, 360, 48
depth = depth[8:, :, :] # cutoff boundary: lat, lon, lev
depth = depth.astype(int)*1.0
depth[depth == 0] = np.nan

class data_generator( Sequence ) :
  def __init__(self, xnames, ynames, level1, level2, batch_size=10) :
    self.xnames = xnames
    self.ynames = ynames
    self.batch_size = batch_size
    self.lowbound = level1 - 1
    self.upbound = level2

  def __len__(self) :
    return np.ceil( len(self.xnames) / float(self.batch_size)).astype(int)

  def __getitem__(self, idx) :

    batch_x = self.xnames[ idx*self.batch_size : idx*self.batch_size + self.batch_size ]
    batch_y = self.ynames[ idx*self.batch_size : idx*self.batch_size + self.batch_size ]

    tempx = np.stack([ np.load(s)['x'] for s in batch_x])   # batch_size, 10, 56, 360
    tempy = np.stack([ np.load(s)['y'] for s in batch_y])

    # batch_size, 56, 360, 10
    tempx = np.moveaxis(tempx, 1, -1)
    tempy = np.moveaxis(tempy, 1, -1)

    tempy[np.where(tempy <= 0 )] = np.nan
    depth_mask = np.repeat(depth[np.newaxis, :, :, :], tempy.shape[0], 0)
    tempy = tempy[:, :, :, self.lowbound:self.upbound]*depth_mask[:, :, :, self.lowbound:self.upbound]

    tempy[np.where(np.isnan(tempy))] = 0
    tempx = process_x(tempx)

    # add buffer domain
    tempx = add_buffer(tempx)
    tempy = add_buffer(tempy)

    return tempx, tempy
