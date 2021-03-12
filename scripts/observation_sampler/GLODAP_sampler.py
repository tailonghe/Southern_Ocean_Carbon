


import netCDF4 as nc
import glob
import numpy as np
import datetime
import pandas as pd


df = pd.read_csv('GLODAPv2.2020_Merged_Master_File.csv')
df = df.loc[ (df['year'] >= 1998) & (df['year'] <= 2019) & (df['latitude'] <= -28) & (df['tco2'] > 0) ]



z = np.array([-2.1, -6.7, -12.15, -18.55, -26.25, -35.25, -45, -55, -65, -75, -85,
    -95, -105, -115, -125, -135, -146.5, -161.5, -180, -200, -220, -240,
    -260, -280, -301, -327, -361, -402.5, -450, -500, -551.5, -614, -700,
    -800, -900, -1000, -1100, -1225, -1400, -1600, -1800, -2010, -2270,
    -2610, -3000, -3400, -3800, -4200, -4600, -5000, -5400, -5800])
z = -z
z = z[:48]


for yy in range(1998, 2020):
    ids = np.genfromtxt('%04dids.txt'%yy, dtype='str')
    idsize = len(ids)

    ynow = np.empty((idsize, 1, 48))
    ynow[:] = -999
    ygeotime = np.zeros((idsize, 3))


    for iid in range(idsize):
        snow = ids[iid]
        # print('Station now: ', snow)
        year = float(snow.split('-')[0])
        cc = float(snow.split('-')[1])
        ss = float(snow.split('-')[2])
        subdfnow = df.loc[ (df['year'] == year) & (df['station'] == ss) & (df['cruise'] == cc) ]

        latnow = list(set(subdfnow['latitude']))
        lonnow = list(set(subdfnow['longitude']))
        if len(latnow) > 1:
            latnow = np.mean(latnow)
        elif len(latnow) == 1:
            latnow = latnow[0]
        else:
            raise Exception

        if len(lonnow) > 1:
            lonnow = np.mean(lonnow)
        elif len(lonnow) == 1:
            lonnow = lonnow[0]
        else:
            raise Exception

        tnow = datetime.datetime(year=int(list(set(subdfnow['year']))[0]), month=int(list(set(subdfnow['month']))[0]), 
                                        day=int(list(set(subdfnow['day']))[0]), hour=int(list(set(subdfnow['hour']))[0]),
                                 minute=int(list(set(subdfnow['minute']))[0]))

        dicnow = subdfnow['tco2'].values
        depthnow = subdfnow['depth'].values

        depthnow = depthnow[ np.where(dicnow > 0) ]
        dicnow = dicnow[ np.where(dicnow > 0) ]

        for jj in range(48):

            if np.any(depthnow == z[jj]):
                subnow = dicnow[ np.where(depthnow == z[jj]) ]
                if np.any(subnow < 0):
                    raise Exception
                ynow[iid, 0, jj] = np.mean(subnow)

            elif np.any(depthnow > z[jj]) and np.any(depthnow < z[jj]):
                deeper = depthnow[ np.where(depthnow > z[jj]) ]
                shallower = depthnow[ np.where(depthnow < z[jj]) ]

                deepind = np.where( depthnow == deeper[np.argmin( abs(deeper - z[jj]))] )
                shallowind = np.where( depthnow == shallower[np.argmin( abs(shallower - z[jj]) )])

                if np.any(dicnow[deepind] < 0) or np.any(dicnow[shallowind] < 0) or np.any(depthnow[deepind] < 0) or \
                                                                     np.any(depthnow[shallowind] < 0)     :
                    print(dicnow[deepind], dicnow[shallowind], depthnow[deepind], depthnow[shallowind])
                    raise Exception

                xa = np.mean(depthnow[deepind])
                xb = np.mean(depthnow[shallowind])
                ya = np.mean(dicnow[deepind])
                yb = np.mean(dicnow[shallowind])

                # print(z[jj], depthnow[deepind], depthnow[shallowind])

                ynow[iid, 0, jj] = ya + (z[jj] - xa)*( yb - ya )/( xb - xa )
        # print('ynow: ', ynow[iid, 0, :])

    np.save('bottle_Y_raw/%04d_smoothed.npy'%year, ynow)
    ynow[ np.where(ynow < 0) ] = np.nan
    print('Now saving: ', 'Y/%04d_smoothed.npy'%year, ynow.shape, np.nanmean(ynow))
    
