import pandas as pd
import numpy as np
import datetime
import netCDF4 as nc
import glob


z = np.array([-2.1, -6.7, -12.15, -18.55, -26.25, -35.25, -45, -55, -65, -75, -85,
    -95, -105, -115, -125, -135, -146.5, -161.5, -180, -200, -220, -240,
    -260, -280, -301, -327, -361, -402.5, -450, -500, -551.5, -614, -700,
    -800, -900, -1000, -1100, -1225, -1400, -1600, -1800, -2010, -2270,
    -2610, -3000, -3400, -3800, -4200, -4600, -5000, -5400, -5800])
z = -z
z = z[:48]

zbounds = np.array([0] + list((z[1:] + z[:-1])/2) + [4400])

eralons = np.arange(0.5, 360.5, 1)
eralats = np.arange(-32.5, -80.5, -1)
print(eralats)
print(eralons)

def find_nearest(value, array):
    idx = (np.abs(array-value)).argmin()
    return idx

def find_k(value, array):
    ind = 0
    while value > array[ind]:
        ind += 1
    return ind-1


flist = glob.glob('ArgoFloatData/*.txt')


for yy in range(2014, 2020):
    obsnow = np.zeros((73, 48, 360, 48))
    counts = np.zeros((73, 48, 360, 48))  # time, lat, lon, depth

    base = datetime.datetime(yy, 1, 1)
    stime = np.array([base + datetime.timedelta(days=x) for x in range(0, 365, 5)])
    # print(stime)
    etime = stime + datetime.timedelta(days=5)
    print(etime.shape)

    for _file in flist:
        idnow = _file.split('/')[-1].split('.')[-2]
        print(idnow)

        df = pd.read_csv(_file, skiprows=5)
        df = df.loc[(df['Date/GMT'].str.contains('%04d'%yy)) &  (df[' DIC_LIAR[µMOL/KG] '] > 0) 
        &  (df['LAT [°N]'] <= -32) &  (df['LAT [°N]'] >= -80)]   # select current year only

        for index, row in df.iterrows():
            #print(row['Date/GMT'], row['LON [°E] '], row['LAT [°N]'], row['DEPTH[M] '], row[' DIC_LIAR[µMOL/KG] '])
            _time = datetime.datetime.strptime(row['Date/GMT'], "%m/%d/%Y %H:%M")
            _lat = row['LAT [°N]']
            _lon = row['LON [°E] ']
            _depth = -row['DEPTH[M] ']
            _dic = row[' DIC_LIAR[µMOL/KG] ']
            kk = find_k(_depth, zbounds)
            ii = find_nearest(_lon, eralons)
            jj = find_nearest(_lat, eralats)
            tidx = np.logical_and( stime<=_time, etime>_time)
            # print(_time, stime[tidx], _lon, eralons[ii], _lat, eralats[jj], _depth, zbounds[kk], zbounds[kk+1], _dic)

            obsnow[tidx, jj, ii, kk] += _dic
            counts[tidx, jj, ii, kk] += 1


    counts[np.where(counts == 0)] = np.nan
    obsnow = obsnow/counts
    print('year: ', yy, '  ', np.nansum(counts))
    for _tidx in range(73):
        if np.nansum(counts[_tidx, :, :, :]) == np.nansum(counts[_tidx, :, :, :]):
            np.savez('Argo_sampled/sampled_Argo_%04d_%02d.npz'% (yy, _tidx), mean=obsnow[_tidx, :, :, :], counts=counts[_tidx, :, :, :], 
                lat=eralats, lon=eralons, level=z)

    np.savez('Argo_sampled/year_merged/sampled_GLODAPv2_%04d.npz'%yy, mean=obsnow[:, :, :, :], counts=counts[:, :, :, :], 
        lat=eralats, lon=eralons, level=z)