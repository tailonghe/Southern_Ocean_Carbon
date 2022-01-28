import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import glob



yy = 2020
flist = glob.glob('ArgoFloatData/*.txt')

for fnow in flist:

    idnow = fnow.split('/')[-1].split('.')[-2]
    print(idnow)
    
    df = pd.read_csv(fnow, skiprows=5)
    df = df[df['Date/GMT'].str.contains('%04d'%yy)]   # select current year only

    doy = df[' Days since 1/1/1900']
    dates = list(set(doy))
    
    curr = 0
    for i in range(len(dates)):
        
        if curr > 9999:
            raise Exception

        subnow = df.loc[ (df[' Days since 1/1/1900'] == dates[i])]
        if  subnow.shape[0] == 0 :
            pass
        else:
            # print(subnow['LON [°E] '].shape, subnow['LAT [°N]'].shape, subnow['DEPTH[M] '].shape, subnow[' DIC_LIAR[µMOL/KG] '].shape)
            # print(subnow[' Days since 1/1/1900'].shape)
            ynow = np.empty((5, subnow['LON [°E] '].shape[0]))
            ynow[0, :] = subnow[' Days since 1/1/1900'].values
            ynow[1, :] = subnow['LON [°E] '].values
            ynow[2, :] = subnow['LAT [°N]'].values
            ynow[3, :] = subnow['DEPTH[M] '].values
            ynow[4, :] = subnow[' DIC_LIAR[µMOL/KG] '].values
            
            # print(ynow.shape)
            
            print('Now saving: ', 'Argo_Y_new_raw/%04d/'%yy + idnow + '_%04d_%04d.npy'% (yy, curr))
            np.save('Argo_Y_new_raw/%04d/'%yy + idnow + '_%04d_%04d.npy'% (yy, curr), ynow) 
            curr += 1

