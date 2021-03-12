import netCDF4 as nc
import glob
import numpy as np
import datetime
from os import path

t0 = datetime.datetime(year=1950, month=1, day=1)

tend = datetime.datetime(year=2020, month=1, day=1)


def fname(fn):
    newf = fn
    newf = 'Argo_X_new/' + newf[15:]
    return newf

def convert(lons):
    """ 0~360 to -180~180
    """
    if lons > 180:
        lons = lons - 360
    return lons


def helper(tdiff):
    ttind = np.argmin(abs(tdiff))
    if tdiff[ttind] > 0:
        ttind = ttind - 1

    if ttind < 0:
        ttind = 0

    return ttind

def nearest_CHL(array2d, xx, yy, xind, yind):

    xv = np.array([list(xx),]*yy.shape[0]).transpose()
    yv = np.array([list(yy),]*xx.shape[0])

    xv = xv - xx[xind]
    yv = yv - yy[yind]
    distances = np.sqrt(xv**2 + yv**2)

    mask = array2d.copy()

    mask[~np.isnan(mask)] = 1
    distances = distances*mask
    xnow, ynow = np.where(distances == np.nanmin(distances))
    
    if np.nanmin(distances) > 1.0:
        final = 0.0001
    else:
        final = array2d[xnow, ynow]
    return np.mean(final), xnow[0], ynow[0]


def nearest(array2d, xx, yy, xind, yind):

    xv = np.array([list(xx),]*yy.shape[0]).transpose()
    yv = np.array([list(yy),]*xx.shape[0])

    xv = xv - xx[xind]
    yv = yv - yy[yind]
    distances = np.sqrt(xv**2 + yv**2)

    mask = np.empty(( array2d.shape[0], array2d.shape[1]  ))
    mask[:] = np.nan
    mask[ np.where(~np.isnan(array2d)) ] = 1

    distances = distances*mask
    xnow, ynow = np.where(distances == np.nanmin(distances))

    final = array2d[xnow, ynow]

    if np.nanmin(distances) > 1:
        return np.nan, xnow[0], ynow[0]
    else:
        return np.mean(final), xnow[0], ynow[0]


def get_ssh(tnow, latnow, lonnow2):
    sshfile = glob.glob('CMEMS/%04d/*_global_allsat_phy_l4_%04d%02d%02d*.nc' % (tnow.year, tnow.year, tnow.month, tnow.day))[0]
    # print('SSH file now: ', sshfile)
    fh = nc.Dataset( sshfile )
    llat = fh.variables['latitude'][:]
    llon = fh.variables['longitude'][:]
    latind = np.argmin(abs(llat - latnow))
    lonind = np.argmin(abs(llon - lonnow2))
    datanow = fh.variables['sla'][0, :, :]
    datanow = np.ma.filled(datanow.astype(float), np.nan)
    datanow, latind1, lonind1 = nearest(datanow, llat, llon, latind, lonind)
    fh.close()
    return datanow, llat[latind1], llon[lonind1]

def get_w(tnow, tprev, latnow, lonnow2):
    # print('Calc W: ', tnow, tprev)
    sshfile1 = glob.glob('CMEMS/%04d/*_global_allsat_phy_l4_%04d%02d%02d*.nc' % (tnow.year, tnow.year, tnow.month, tnow.day))[0]
    fh = nc.Dataset( sshfile1 )
    llat = fh.variables['latitude'][:]
    llon = fh.variables['longitude'][:]
    latind = np.argmin(abs(llat - latnow))
    lonind = np.argmin(abs(llon - lonnow2))
    datanow = fh.variables['sla'][0, :, :]
    datanow = np.ma.filled(datanow.astype(float), np.nan)
    datanow, latind2, lonind2 = nearest(datanow, llat, llon, latind, lonind)
    fh.close()

    sshfile2 = glob.glob('CMEMS/%04d/*_global_allsat_phy_l4_%04d%02d%02d*.nc' % (tprev.year, tprev.year, tprev.month, tprev.day))[0]
    fh = nc.Dataset( sshfile2 )
    dataprev = fh.variables['sla'][0, :, :]
    dataprev = np.ma.filled(dataprev.astype(float), np.nan)
    dataprev, latind2, lonind2 = nearest(dataprev, llat, llon, latind, lonind)
    fh.close()

    w = (datanow - dataprev) / 86400.0 # daily W

    return w, llat[latind2], llon[lonind2]

def pair_now(taryear):

    flist = glob.glob( 'Argo_Y_new_raw/%04d/*npy' % taryear )

    xnow = np.empty((len(flist), 1, 10))
    xnow[:] = np.nan

    for ind in range(len(flist)):

        s = flist[ind]
        print(ind, ' / ', len(flist))
        print('track now: ', s.split('/')[-1][:-3])

        fnow = np.load(s)
        
        doy = fnow[0, 0]
        latnow = fnow[2, 0]
        lonnow2 = fnow[1, 0]
        lonnow = convert(lonnow2)  #  (-180 to 180)

        tnow = datetime.datetime(year=1900, month=1, day=1) + datetime.timedelta(days=doy)

        if tnow >= tend:
            print('Passing: ', tnow)
            pass
        else:
            year = tnow.year

            print('=======================================================')
            print(tnow, lonnow, lonnow2, latnow)

            slamean = 0
            slac = 0
            wmean = 0
            wc = 0

            for backsteps in range(0, 5): # 5 days backwards
                dt = datetime.timedelta(days= int(backsteps))
                dt2 = datetime.timedelta(days= int(backsteps + 1))
                slatemp, latind1, lonind1 = get_ssh(tnow-dt, latnow, lonnow2)
                wtemp, latind2, lonind2 = get_w(tnow-dt, tnow-dt2, latnow, lonnow2)
                if ~np.isnan(slatemp):
                    slamean += slatemp
                    slac += 1
                if ~np.isnan(wtemp):
                    wmean += wtemp
                    wc += 1

            if slac == 0:
                slamean = np.nan
            else:
                slamean = slamean / slac

            if wc == 0:
                wmean = np.nan
            else:
                wmean = wmean / wc
            
             # SLA(Time, Longitude, Latitude), Time(Days since 1985-01-01 00:00:00), Latitude, Longitude(0-360), 5-day, meter
            # sshfile = glob.glob('CMEMS/%04d/*_global_allsat_phy_l4_%04d%02d%02d*.nc'%(tnow.year, tnow.year, tnow.month, tnow.day))[0]
            # print('SSH file now: ', sshfile)
            # fh = nc.Dataset( sshfile )
            # llat = fh.variables['latitude'][:]
            # llon = fh.variables['longitude'][:]
            # latind = np.argmin(abs(llat - latnow))
            # lonind = np.argmin(abs(llon - lonnow2))
            # datanow = fh.variables['sla'][0, :, :]
            # datanow = np.ma.filled(datanow.astype(float), np.nan)
            # datanow, latind1, lonind1 = nearest(datanow, llat, llon, latind, lonind)
            # fh.close()

            # tprev = tnow - datetime.timedelta(days=5)
            # sshfile = glob.glob('CMEMS/%04d/*_global_allsat_phy_l4_%04d%02d%02d*.nc' % (tprev.year, tprev.year, tprev.month, tprev.day))[0]
            # print('SSH file now: ', sshfile)
            # fh = nc.Dataset( sshfile )
            # llat = fh.variables['latitude'][:]
            # llon = fh.variables['longitude'][:]
            # latind = np.argmin(abs(llat - latnow))
            # lonind = np.argmin(abs(llon - lonnow2))
            # dataprev = fh.variables['sla'][0, :, :]
            # dataprev = np.ma.filled(dataprev.astype(float), np.nan)
            # dataprev, latind2, lonind2 = nearest(dataprev, llat, llon, latind, lonind)
            # fh.close()

            # print(datanow, dataprev)
            xnow[ind, 0, 0] = slamean 
            print('SSHA(', lonind1, latind1, "): ", xnow[ind, 0, 0])
            xnow[ind, 0, 5] = wmean
            print('Wvel(', lonind2, latind2, "): ", xnow[ind, 0, 5])

            # pco2(time, lat, lon), time(month), lat, lon(-180-180), Clim, muatm
            pCO2file = 'spco2_MPI-SOM_FFN_v2020.nc'
            print('pCO2 file now: ', pCO2file)
            fh = nc.Dataset( pCO2file )
            timelist = fh.variables['time'][:]
            timelist = np.array([datetime.timedelta(seconds=int(s)) for s in timelist])
            timelist = timelist + datetime.datetime(year=2000, month=1, day=1)
            pco2dt = np.array([s.year * 100 + s.month for s in timelist])
            ttind = np.where(pco2dt == tnow.year * 100 + tnow.month)[0][0]
            llat = fh.variables['lat'][:]
            llon = fh.variables['lon'][:]
            latind = np.argmin(abs(llat - latnow))
            lonind = np.argmin(abs(llon - lonnow))
            datanow = fh.variables['spco2_smoothed'][ttind, :, :]
            datanow[np.where(datanow > 1e10) ] = np.nan
            datanow, latind, lonind = nearest(datanow, llat, llon, latind, lonind)
            xnow[ind, 0, 1] = datanow
            print('pCO2(', timelist[ttind], llon[lonind], llat[latind], "): ", xnow[ind, 0, 1])
            fh.close()


            # sshf(time, latitude, longitude), latitude(-180-180), time(hours since 1900-01-01 00:00:00.0), hourly, W/m^2
            tflxfile = '../SOCO2_phase2/Tflx/%04d/sea_surface_heatflux_%04d_%02d.nc' % (tnow.year, tnow.year, tnow.month)
            print('tflux file now: ', tflxfile)
            fh = nc.Dataset( tflxfile )
            timelist = fh.variables['time'][:]
            timelist = np.array([datetime.timedelta(hours=int(s)) for s in timelist])
            timelist = timelist + datetime.datetime(year=1900, month=1, day=1)
            tdiff = timelist - tnow
            tdiff = np.array([xxx.total_seconds() for xxx in tdiff])
            ttind = helper(tdiff)
            llat = fh.variables['latitude'][:]
            llon = fh.variables['longitude'][:]
            latind = np.argmin(abs(llat - latnow))
            lonind = np.argmin(abs(llon - lonnow))
            datanow = fh.variables['slhf'][:] + fh.variables['sshf'][:]  + fh.variables['ssr'][:]  + fh.variables['str'][:]
            datanow = datanow[ttind, latind, lonind ]
            xnow[ind, 0, 2] = datanow / 3600.  # convert to W m^-2
            print('SSHF(', llon[lonind], llat[latind], "): ", xnow[ind, 0, 2])
            fh.close()


            # u/v(time, depth, latitude, longitude), latitude, longitude(20-420), time(Day since 1992-10-05 00:00:00), 5-day, m/s
            uvfile = '../SOCO2_phase2/OSCAR/oscar_vel%04d.nc' % year
            # print('uv file now: ', uvfile)
            fh = nc.Dataset( uvfile )
            timelist = fh.variables['time'][:]
            timelist = np.array([datetime.timedelta(days=int(s)) for s in timelist])
            timelist = timelist + datetime.datetime(year=1992, month=10, day=5)
            tdiff = timelist - tnow
            tdiff = np.array([xxx.total_seconds() for xxx in tdiff])
            ttind = helper(tdiff)
            llat = fh.variables['latitude'][:]
            llon = fh.variables['longitude'][:]
            llon[np.where(llon > 360)] = llon[np.where(llon > 360)] - 360
            latind = np.argmin(abs(llat - latnow))
            lonind = np.argmin(abs(llon - lonnow2))

            datanow = fh.variables['u'][ttind, 0, :, :]
            datanow, latind, lonind = nearest(datanow, llat, llon, latind, lonind)
            xnow[ind, 0, 3] = datanow
            print('Uvel(', llon[lonind], llat[latind], "): ", xnow[ind, 0, 3])
            datanow = fh.variables['v'][ttind, 0, :, :]
            datanow, latind, lonind = nearest(datanow, llat, llon, latind, lonind)
            xnow[ind, 0, 4] = datanow
            print('Vvel(', llon[lonind], llat[latind], "): ", xnow[ind, 0, 4])
            fh.close()

            # chlor_a(lat, lon), lat, lon(-180, 180), DOY, daily, mg/m^3
            # day_of_year = (tnow - datetime.datetime(tnow.year, 1, 1)).days + 1
            chlfile = glob.glob('../SOCO2_phase2/CHL/monthly/L3m_%04d%02d*%04d%02d*4_GSM*'%(year,tnow.month,year,tnow.month ))[0]
            print('chlfile now: ', chlfile)
            fh = nc.Dataset( chlfile )
            llat = fh.variables['lat'][:]
            llon = fh.variables['lon'][:]
            latind = np.argmin(abs(llat - latnow))
            lonind = np.argmin(abs(llon - lonnow))
            datanow = fh.variables['CHL1_mean'][:]
            datanow = np.ma.filled(datanow.astype(float), np.nan)
            datanow, latind, lonind = nearest_CHL(datanow, llat, llon, latind, lonind)
            xnow[ind, 0, 6] = datanow
            print('CHL-a(', llon[lonind], llat[latind], "): ", xnow[ind, 0, 6])
            fh.close()

            # u10/v10 (time, latitude, longitude), latitude, longitude(-180 -- 180), time(hours since 1900-01-01 00:00:00.0)
            windfile = '../ERA5/%04d/sea_surface_PTUV_%04d_%02d.nc' % ( tnow.year, tnow.year, tnow.month )
            print('windfile now: ', windfile)
            fh = nc.Dataset( windfile )
            timelist = fh.variables['time'][:]
            timelist = np.array([datetime.timedelta(hours=int(s)) for s in timelist])
            timelist = timelist + datetime.datetime(year=1900, month=1, day=1)
            tdiff = timelist - tnow
            tdiff = np.array([xxx.total_seconds() for xxx in tdiff])
            ttind = helper(tdiff)
            llat = fh.variables['latitude'][:]
            llon = fh.variables['longitude'][:]
            latind = np.argmin(abs(llat - latnow))
            lonind = np.argmin(abs(llon - lonnow))

            datanow = fh.variables['sst'][:]
            datanow = datanow[ttind, :, : ]
            datanow =  np.ma.filled(datanow.astype(float), np.nan)
            datanow, latind1, lonind1 = nearest(datanow, llat, llon, latind, lonind)
            xnow[ind, 0, 7] = datanow

            datanow = fh.variables['u10'][:]
            datanow = datanow[ttind, latind, lonind ]
            xnow[ind, 0, 8] = datanow

            datanow = fh.variables['v10'][:]
            datanow = datanow[ttind, latind, lonind ]
            xnow[ind, 0, 9] = datanow
            fh.close()

            print('sst( ', llon[lonind1], llat[latind1], "): ", xnow[ind, 0, 7])
            print('u10m(', llon[lonind], llat[latind], "): ", xnow[ind, 0, 8])

            print('v10m(', llon[lonind], llat[latind], "): ", xnow[ind, 0, 9])

            print('=======================================================')

            outfn = fname(s)
            print('NOW saving: ', outfn)
            np.save(outfn, xnow[ind, :, :])


        print('NOW saving X: ', '%04d_x_predictors.npy' % taryear)
        np.save('Argo_X_raw/' + '%04d_x_predictors.npy'% taryear, xnow)



if __name__ == '__main__':
    for yy in range(2014, 2020):
        pair_now(yy)
