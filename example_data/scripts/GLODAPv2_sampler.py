import pandas as pd
import numpy as np
import datetime
import netCDF4 as nc

df = pd.read_csv('GLODAPv2.2021_Merged_Master_File.csv')
subdf = df.loc[ (df['G2year'] >= 1993) & (df['G2year'] <= 2019) & (df['G2latitude'] <= -32) 
               & (df['G2tco2'] > 0) & (df['G2depth'] < 4400) & (df['G2latitude'] >= -80) ]
lons = subdf['G2longitude']
lats = subdf['G2latitude']


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


alldts = pd.to_datetime(subdf.G2year*10000+subdf.G2month*100+subdf.G2day, format='%Y%m%d')
hours = pd.to_timedelta(subdf['G2hour'], unit='h')
minutes = pd.to_timedelta(subdf['G2minute'], unit='m')
alldts = alldts + hours + minutes
print(alldts.min(), alldts.max())

def find_nearest(value, array):
    idx = (np.abs(array-value)).argmin()
    return idx

def find_k(value, array):
    ind = 0
    while value > array[ind]:
        ind += 1
        
    return ind-1


def find_index(array1, array2, function):
    out = np.zeros((len(array1)))
    
    if len(array1) == 0:
        return -1
    else:
        for i in range(len(array1)):
            out[i] = function(array1[i], array2)

        return out


for yy in range(1993, 2020):
    print('year: ', yy)
    stime = datetime.datetime(yy, 1, 1)
    obsnow = np.zeros((73, 48, 360, 48))
    counts = np.zeros((73, 48, 360, 48))  # time, lat, lon, depth
    curr = 0

    while stime < datetime.datetime(yy, 12, 31):

        etime = stime + datetime.timedelta(days=5)
        subdfnow = subdf.loc[ (alldts >= stime) & (alldts < etime) ]
        latnow = subdfnow.G2latitude
        lonnow = subdfnow.G2longitude
        lonnow = lonnow%360.0
        znow = subdfnow.G2depth
        dicnow = subdfnow.G2tco2

        if len(znow) == 0:
            obsnow[curr] = np.nan
            counts[curr] = np.nan
        else:
    #         print(len(znow), len(zbounds), np.max(znow))
            know = find_index(znow.values, zbounds, find_k).astype(int)
            inow = find_index(lonnow.values, eralons, find_nearest).astype(int)
            jnow = find_index(latnow.values, eralats, find_nearest).astype(int)

            for jj in range(len(know)):
                if know[jj] > 47:
                    pass
                else:
    #                 print('i, j, k: ', inow[jj], jnow[jj], know[jj], znow.values[jj])
                    obsnow[curr, jnow[jj], inow[jj], know[jj]] += dicnow.values[jj]
                    counts[curr, jnow[jj], inow[jj], know[jj]] += 1

        curr += 1
        stime = etime

    obsnow = obsnow/counts
    print('year: ', yy, '  ', np.nansum(counts))

    for tidx in range(73):
        if np.nansum(counts[tidx, :, :, :]) == np.nansum(counts[tidx, :, :, :]):
            np.savez('GLODAPv2_sampled/sampled_GLODAPv2_%04d_%02d.npz'% (yy, tidx), mean=obsnow[tidx, :, :, :], counts=counts[tidx, :, :, :], 
                lat=eralats, lon=eralons, level=z)       

    np.savez('GLODAPv2_sampled/year_merged/sampled_GLODAPv2_%04d.npz'%yy, mean=obsnow[:, :, :, :], counts=counts[:, :, :, :], 
    	lat=eralats, lon=eralons, level=z)