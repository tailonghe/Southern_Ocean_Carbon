import netCDF4 as nc
import glob
import numpy as np
import datetime
import pandas as pd


def helper(tdiff):
    ttind = np.argmin(abs(tdiff))
    if tdiff[ttind] > 0:
        ttind = ttind - 1

    if ttind < 0:
        ttind = 0

    return ttind

def find_chl_file(dt):
    chllist = glob.glob('../SOCO2_phase2/CHL/8day/L3m_%04d*.nc_1x1' % dt.year)

    if len(chllist) == 0:
        return None

    slist = np.array([int(s.split('/')[-1][8:12]) for s in chllist])
    elist = np.array([int(s.split('/')[-1][17:21]) for s in chllist])

    dtnow = dt.month * 100 + dt.day
    loc = np.where( np.logical_and( slist <= dtnow, elist >= dtnow) )[0]
    if len(loc) == 0:
        return None
    else:
        return chllist[loc[0]]
    

for yy in range(1998, 2020):
    xnow = np.zeros((73, 56, 360, 10, 5))
    xnow[:] = np.nan

    tnow = datetime.datetime(yy, 1, 1)
    tend = datetime.datetime(yy+1, 1, 1)
    
    curr = 0
    while tnow < tend and curr <= 72:  
        for tc in range(5):
            subt = tnow + datetime.timedelta(days=tc)
            print(subt)

            fnow = find_chl_file(subt)
            if fnow == None:
                pass
            else:

                # SSHA, W
                fnow = glob.glob('../Phase2_clean/CMEMS/%04d/*_global_allsat_phy_l4_%04d%02d%02d_*.nc_1x1'%(subt.year, subt.year, subt.month, subt.day))[0]
                fh = nc.Dataset(fnow)
                print('SSHA file: ', fnow)
                datanow = fh.variables['sla'][0]  # -180~180
                datanow = np.ma.filled(datanow.astype(float), np.nan)
                fh.close()
                xnow[curr, :, :, 0, tc] = datanow

                # pCO2
                fnow = '../Phase2_clean/spco2_MPI-SOM_FFN_v2020.nc'
                fh = nc.Dataset(fnow)
                timelist = fh.variables['time'][:]
                timelist = np.array([datetime.timedelta(seconds=int(s)) for s in timelist])
                timelist = timelist + datetime.datetime(year=2000, month=1, day=1)
                pco2dt = np.array([s.year * 100 + s.month for s in timelist])
                ttind = np.where(pco2dt == subt.year * 100 + subt.month)[0][0]
                datanow = fh.variables['spco2_smoothed'][ttind]
                datanow[np.where(datanow > 1e10) ] = np.nan
                fh.close()
                datanow = datanow[9:65]
                datanow = datanow[::-1, :]
                xnow[curr, :, :, 1, tc] = datanow

                #, Tflx
                fnow = glob.glob('../SOCO2_phase2/Tflx/%04d/sea_surface_heatflux_%04d_%02d_1x1.nc'%(subt.year, subt.year, subt.month))[0]
                print(fnow)
                fh = nc.Dataset(fnow)
                timelist = fh.variables['time'][:]
                timelist = np.array([datetime.timedelta(hours=int(s)) for s in timelist])
                timelist = timelist + datetime.datetime(year=1900, month=1, day=1)
                tdiff = timelist - subt
                tdiff = np.array([xxx.total_seconds() for xxx in tdiff])
                ttind = helper(tdiff)
                print('Tflx file: ', fnow)
                datanow = np.nanmean( fh.variables['ssr'][ttind:ttind+24, :, :] , axis=0)   # -180~180
                datanow = datanow + np.nanmean( fh.variables['str'][ttind:ttind+24, :, :] , axis=0)   # -180~180
                datanow = datanow + np.nanmean( fh.variables['slhf'][ttind:ttind+24, :, :] , axis=0)   # -180~180
                datanow = datanow + np.nanmean( fh.variables['sshf'][ttind:ttind+24, :, :] , axis=0)   # -180~180
                fh.close()
                xnow[curr, :, :, 2, tc] = datanow / 3600.0 # convert to W / m^2


                #, u, v
                fnow = '../SOCO2_phase2/OSCAR/oscar_vel%04d_1x1.nc' % subt.year
                fh = nc.Dataset(fnow)
                timelist = fh.variables['time'][:]
                timelist = np.array([datetime.timedelta(days=int(s)) for s in timelist])
                timelist = timelist + datetime.datetime(year=1992, month=10, day=5)
                tdiff = timelist - subt
                tdiff = np.array([xxx.total_seconds() for xxx in tdiff])
                ttind = helper(tdiff)

                datanow = fh.variables['u'][ttind, 0, :, :]    # -180~180
                datanow = np.ma.filled(datanow.astype(float), np.nan)
                datanow = np.roll(datanow, 200, axis=-1)
                xnow[curr, :, :, 3, tc] = datanow

                datanow = fh.variables['v'][ttind, 0, :, :]    # -180~180
                datanow = np.ma.filled(datanow.astype(float), np.nan)
                datanow = np.roll(datanow, 200, axis=-1)
                xnow[curr, :, :, 4, tc] = datanow

                #, CHL
                fnow = find_chl_file(subt)
                print('CHL file: ', fnow)
                fh = nc.Dataset(fnow)
                datanow = fh.variables['CHL1_mean'][:]
                datanow = np.ma.filled(datanow.astype(float), np.nan)
                datanow[np.where(np.isnan(datanow)) ] = 0.0001
                xnow[curr, :, :, 6, tc] = datanow

                # , SST, u10, v10
                fnow = glob.glob('../ERA5/%04d/sea_surface_PTUV_%04d_%02d.1x1.nc'%(subt.year, subt.year, subt.month))[0]
                fh = nc.Dataset(fnow)
                timelist = fh.variables['time'][:]
                timelist = np.array([datetime.timedelta(hours=int(s)) for s in timelist])
                timelist = timelist + datetime.datetime(year=1900, month=1, day=1)
                tdiff = timelist - subt
                tdiff = np.array([xxx.total_seconds() for xxx in tdiff])
                ttind = helper(tdiff)


                datanow = np.nanmean( fh.variables['sst'][ttind:ttind+24, :, :] , axis=0 ) 
                # print(fnow, fh.variables['sst'], datanow.shape)
                xnow[curr, :, :, 7, tc] = datanow

                datanow = np.nanmean( fh.variables['u10'][ttind:ttind+24, :, :] , axis=0 ) 
                xnow[curr, :, :, 8, tc] = datanow

                datanow = np.nanmean( fh.variables['v10'][ttind:ttind+24, :, :] , axis=0 ) 
                xnow[curr, :, :, 9, tc] = datanow

                fh.close()

        tnow = tnow + datetime.timedelta(days=5)
        curr += 1



    # (73, 56, 360, 10, 5)
    for i in range(5):
        # calculate 5-day average W
        xnow[:, :, :, 5, i] = xnow[:, :, :, 0, -1] - xnow[:, :, :, 0, 0]

    xnow = np.nanmean(xnow, axis=-1)
    print(xnow.shape)

    np.save('X_predictors_%04d.npy' % yy, xnow)
