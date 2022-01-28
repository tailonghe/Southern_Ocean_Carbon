import netCDF4 as nc
import glob
import numpy as np
import datetime
import pandas as pd


z = np.array([-2.1, -6.7, -12.15, -18.55, -26.25, -35.25, -45, -55, -65, -75, -85,
    -95, -105, -115, -125, -135, -146.5, -161.5, -180, -200, -220, -240,
    -260, -280, -301, -327, -361, -402.5, -450, -500, -551.5, -614, -700,
    -800, -900, -1000, -1100, -1225, -1400, -1600, -1800, -2010, -2270,
    -2610, -3000, -3400, -3800, -4200, -4600, -5000, -5400, -5800])
z = -z
z = z[:50]

df = pd.read_csv('GLODAPv2.2020_Merged_Master_File.csv')
df = df.loc[ (df['year'] >= 1998) & (df['year'] <= 2019) & (df['latitude'] <= -28) & (df['tco2'] > 0) ]


def convert(lons):
    return lons % 360

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
        final = 0
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

    # print('inside nearest: ', xnow, ynow)
    final = array2d[xnow, ynow]

    if np.nanmin(distances) > 1:
        return np.nan, xnow[0], ynow[0]
    else:
        # print(np.nanmin(distances), final, xnow, ynow)
        return np.mean(final), xnow[0], ynow[0]


def get_ssh(tnow, latnow, lonnow2):
    sshfile = glob.glob('CMEMS/%04d/dt_global_allsat_phy_l4_%04d%02d%02d*.nc' % (tnow.year, tnow.year, tnow.month, tnow.day))[0]
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
    sshfile1 = glob.glob('CMEMS/%04d/dt_global_allsat_phy_l4_%04d%02d%02d*.nc' % (tnow.year, tnow.year, tnow.month, tnow.day))[0]
    fh = nc.Dataset( sshfile1 )
    llat = fh.variables['latitude'][:]
    llon = fh.variables['longitude'][:]
    latind = np.argmin(abs(llat - latnow))
    lonind = np.argmin(abs(llon - lonnow2))
    datanow = fh.variables['sla'][0, :, :]
    datanow = np.ma.filled(datanow.astype(float), np.nan)
    datanow, latind2, lonind2 = nearest(datanow, llat, llon, latind, lonind)
    fh.close()

    sshfile2 = glob.glob('CMEMS/%04d/dt_global_allsat_phy_l4_%04d%02d%02d*.nc' % (tprev.year, tprev.year, tprev.month, tprev.day))[0]
    fh = nc.Dataset( sshfile2 )
    dataprev = fh.variables['sla'][0, :, :]
    dataprev = np.ma.filled(dataprev.astype(float), np.nan)
    dataprev, latind2, lonind2 = nearest(dataprev, llat, llon, latind, lonind)
    fh.close()

    w = (datanow - dataprev) / 86400.0 # daily W

    return w, llat[latind2], llon[lonind2]

for yy in range(2019, 2020):
    ids = np.genfromtxt('%04dids.txt'%yy, dtype='str')
    idsize = len(ids)

    xnow = np.zeros((idsize, 1, 10))

    for iid in range(idsize):
        snow = ids[iid]
        print("ID now: ", snow, iid, ' / ', idsize)
        year = float(snow.split('-')[0])
        cc = float(snow.split('-')[1])
        ss = float(snow.split('-')[2])
        subdfnow = df.loc[ (df['year'] == year) & (df['station'] == ss) & (df['cruise'] == cc) ]

        latnow = list(set(subdfnow['latitude']))
        lonnow = list(set(subdfnow['longitude']))
        if len(latnow) != 1:
            raise Exception

        latnow = latnow[0]
        lonnow = lonnow[0]          # -180 to 180
        lonnow2 = convert(lonnow)   # 0 to 360

        tnow = datetime.datetime(year=int(list(set(subdfnow['year']))[0]), month=int(list(set(subdfnow['month']))[0]), 
                                        day=int(list(set(subdfnow['day']))[0]), hour=int(list(set(subdfnow['hour']))[0]),
                                 minute=int(list(set(subdfnow['minute']))[0]))


        dicnow = subdfnow['tco2'].values
        depthnow = subdfnow['depth'].values

        depthnow = depthnow[ np.where(dicnow > 0) ]
        dicnow = dicnow[ np.where(dicnow > 0) ]

        # print(depthnow, dicnow)
        if ((dicnow.shape[0] == 0) or (len(glob.glob('../SOCO2_phase2/CHL/monthly/L3m_%04d%02d*%04d%02d*_4_*'%(year,tnow.month,year,tnow.month ))) < 1)):
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', glob.glob('../SOCO2_phase2/CHL/monthly/L3m_%04d%02d*%04d%02d*_4_*'%(year,tnow.month,year,tnow.month ))) 
            pass
        else:
            # print("ynow: ", ynow[iid, 0, :])
            print('=======================================================')
            print(tnow, lonnow, lonnow2, latnow)
            

            slamean = 0
            slac = 0
            wmean = 0
            wc = 0

            for backsteps in range(0, 5): # 1, 2, 3, 4
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
            xnow[iid, 0, 0] = slamean 
            print('SSHA(', llon[lonind1], llat[latind1], "): ", xnow[iid, 0, 0])
            xnow[iid, 0, 5] = wmean
            print('Wvel(', llon[lonind2], llat[latind2], "): ", xnow[iid, 0, 5])

            # pco2(time, lat, lon), time(month), lat, lon(-180-180), Clim, muatm
            pCO2file = 'spco2_MPI-SOM_FFN_v2020.nc'
            # print('pCO2 file now: ', pCO2file)
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
            datanow = fh.variables['spco2_smoothed'][ttind]
            datanow[np.where(datanow > 1e10) ] = np.nan
            datanow, latind, lonind = nearest(datanow, llat, llon, latind, lonind)
            xnow[iid, 0, 1] = datanow
            print('pCO2(', timelist[ttind], llon[lonind], llat[latind], "): ", xnow[iid, 0, 1])
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
            xnow[iid, 0, 2] = datanow / 3600.  # convert to W m^-2
            print('SSHF(', llon[lonind], llat[latind], "): ", xnow[iid, 0, 2])
            fh.close()


            # u/v(time, depth, latitude, longitude), latitude, longitude(20-420), time(Day since 1992-10-05 00:00:00), 5-day, m/s
            uvfile = '../SOCO2_phase2/OSCAR/oscar_vel%04d.nc' % year
            print('uv file now: ', uvfile)
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
            xnow[iid, 0, 3] = datanow
            print('Uvel(', llon[lonind], llat[latind], "): ", xnow[iid, 0, 3])
            

            datanow = fh.variables['v'][ttind, 0, :, :]
            datanow, latind, lonind = nearest(datanow, llat, llon, latind, lonind)
            xnow[iid, 0, 4] = datanow
            
            print('Vvel(', llon[lonind], llat[latind], "): ", xnow[iid, 0, 4])
            fh.close()
            # wfile from sshfile

            # chlor_a(lat, lon), lat, lon(-180, 180), DOY, daily, mg/m^3
            # day_of_year = (tnow - datetime.datetime(tnow.year, 1, 1)).days + 1
            chlfile = glob.glob('../SOCO2_phase2/CHL/monthly/L3m_%04d%02d*%04d%02d*_4_*'%(year,tnow.month,year,tnow.month ))[0]
            print('chlfile now: ', chlfile)
            fh = nc.Dataset( chlfile )
            llat = fh.variables['lat'][:]
            llon = fh.variables['lon'][:]
            latind = np.argmin(abs(llat - latnow))
            lonind = np.argmin(abs(llon - lonnow))
            datanow = fh.variables['CHL1_mean'][:]
            datanow = np.ma.filled(datanow.astype(float), np.nan)

            datanow, latind, lonind = nearest_CHL(datanow, llat, llon, latind, lonind)
            xnow[iid, 0, 6] = datanow
            print('CHL-a(', llon[lonind], llat[latind], "): ", xnow[iid, 0, 6])
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
            xnow[iid, 0, 7] = datanow

            datanow = fh.variables['u10'][:]
            datanow = datanow[ttind, latind, lonind ]
            xnow[iid, 0, 8] = datanow

            datanow = fh.variables['v10'][:]
            datanow = datanow[ttind, latind, lonind ]
            xnow[iid, 0, 9] = datanow
            fh.close()

            print('sst( ', llon[lonind1], llat[latind1], "): ", xnow[iid, 0, 7])
            print('u10m(', llon[lonind], llat[latind], "): ", xnow[iid, 0, 8])

            print('v10m(', llon[lonind], llat[latind], "): ", xnow[iid, 0, 9])

            print('=======================================================')

    # 2016-1024-000000007

    ynow = np.load('bottle_Y_raw/%04d_smoothed.npy'%yy)
    print(xnow.shape, ynow.shape)
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!! NOW saving X: %04d_predictors.npy'%yy)
    np.save('bottle_X_raw/%04d_predictors.npy'%yy, xnow)


for yy in range(1998, 2000):
    ids = np.genfromtxt('%04dids.txt'%yy, dtype='str')
    idsize = len(ids)

    xnow = np.zeros((idsize, 1, 10))

    for iid in range(idsize):
        snow = ids[iid]
        print("ID now: ", snow, iid, ' / ', idsize)
        year = float(snow.split('-')[0])
        cc = float(snow.split('-')[1])
        ss = float(snow.split('-')[2])
        subdfnow = df.loc[ (df['year'] == year) & (df['station'] == ss) & (df['cruise'] == cc) ]

        latnow = list(set(subdfnow['latitude']))
        lonnow = list(set(subdfnow['longitude']))
        if len(latnow) != 1:
            raise Exception

        latnow = latnow[0]
        lonnow = lonnow[0]          # -180 to 180
        lonnow2 = convert(lonnow)   # 0 to 360

        tnow = datetime.datetime(year=int(list(set(subdfnow['year']))[0]), month=int(list(set(subdfnow['month']))[0]), 
                                        day=int(list(set(subdfnow['day']))[0]), hour=int(list(set(subdfnow['hour']))[0]),
                                 minute=int(list(set(subdfnow['minute']))[0]))


        dicnow = subdfnow['tco2'].values
        depthnow = subdfnow['depth'].values

        depthnow = depthnow[ np.where(dicnow > 0) ]
        dicnow = dicnow[ np.where(dicnow > 0) ]

        # print(depthnow, dicnow)
        if ((dicnow.shape[0] == 0) or (len(glob.glob('../SOCO2_phase2/CHL/monthly/L3m_%04d%02d*%04d%02d*_4_*'%(year,tnow.month,year,tnow.month ))) < 1)):
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', glob.glob('../SOCO2_phase2/CHL/monthly/L3m_%04d%02d*%04d%02d*_4_*'%(year,tnow.month,year,tnow.month ))) 
            pass
        else:
            # print("ynow: ", ynow[iid, 0, :])
            print('=======================================================')
            print(tnow, lonnow, lonnow2, latnow)
            

            # slamean = 0
            # slac = 0
            # wmean = 0
            # wc = 0

            # for backsteps in range(0, 5): # 1, 2, 3, 4
            #     dt = datetime.timedelta(days= int(backsteps))
            #     dt2 = datetime.timedelta(days= int(backsteps + 1))
            #     slatemp, latind1, lonind1 = get_ssh(tnow-dt, latnow, lonnow2)
            #     wtemp, latind2, lonind2 = get_w(tnow-dt, tnow-dt2, latnow, lonnow2)
            #     if ~np.isnan(slatemp):
            #         slamean += slatemp
            #         slac += 1
            #     if ~np.isnan(wtemp):
            #         wmean += wtemp
            #         wc += 1

            # if slac == 0:
            #     slamean = np.nan
            # else:
            #     slamean = slamean / slac

            # if wc == 0:
            #     wmean = np.nan
            # else:
            #     wmean = wmean / wc


            # SLA(Time, Longitude, Latitude), Time(Days since 1985-01-01 00:00:00), Latitude, Longitude(0-360), 5-day, meter
            sshfile = glob.glob('CMEMS/%04d/*_global_allsat_phy_l4_%04d%02d%02d*.nc'%(tnow.year, tnow.year, tnow.month, tnow.day))[0]
            print('SSH file now: ', sshfile)
            fh = nc.Dataset( sshfile )
            llat = fh.variables['latitude'][:]
            llon = fh.variables['longitude'][:]
            latind = np.argmin(abs(llat - latnow))
            lonind = np.argmin(abs(llon - lonnow2))
            datanow = fh.variables['sla'][0, :, :]
            datanow = np.ma.filled(datanow.astype(float), np.nan)
            datanow, latind1, lonind1 = nearest(datanow, llat, llon, latind, lonind)
            fh.close()

            tprev = tnow - datetime.timedelta(days=5)
            sshfile = glob.glob('CMEMS/%04d/*_global_allsat_phy_l4_%04d%02d%02d*.nc' % (tprev.year, tprev.year, tprev.month, tprev.day))[0]
            print('SSH file now: ', sshfile)
            fh = nc.Dataset( sshfile )
            llat = fh.variables['latitude'][:]
            llon = fh.variables['longitude'][:]
            latind = np.argmin(abs(llat - latnow))
            lonind = np.argmin(abs(llon - lonnow2))
            dataprev = fh.variables['sla'][0, :, :]
            dataprev = np.ma.filled(dataprev.astype(float), np.nan)
            dataprev, latind2, lonind2 = nearest(dataprev, llat, llon, latind, lonind)
            fh.close()


            # print(datanow, dataprev)
            xnow[iid, 0, 0] = datanow 
            print('SSHA(', llon[lonind1], llat[latind1], "): ", xnow[iid, 0, 0])
            xnow[iid, 0, 5] = (datanow - dataprev)/432000.0
            print('Wvel(', llon[lonind2], llat[latind2], "): ", xnow[iid, 0, 5])

            # pco2(time, lat, lon), time(month), lat, lon(-180-180), Clim, muatm
            pCO2file = 'spco2_MPI-SOM_FFN_v2020.nc'
            # print('pCO2 file now: ', pCO2file)
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
            datanow = fh.variables['spco2_smoothed'][ttind]
            datanow[np.where(datanow > 1e10) ] = np.nan
            datanow, latind, lonind = nearest(datanow, llat, llon, latind, lonind)
            xnow[iid, 0, 1] = datanow
            print('pCO2(', timelist[ttind], llon[lonind], llat[latind], "): ", xnow[iid, 0, 1])
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
            xnow[iid, 0, 2] = datanow / 3600.  # convert to W m^-2
            print('SSHF(', llon[lonind], llat[latind], "): ", xnow[iid, 0, 2])
            fh.close()


            # u/v(time, depth, latitude, longitude), latitude, longitude(20-420), time(Day since 1992-10-05 00:00:00), 5-day, m/s
            uvfile = '../SOCO2_phase2/OSCAR/oscar_vel%04d.nc' % year
            print('uv file now: ', uvfile)
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
            xnow[iid, 0, 3] = datanow
            print('Uvel(', llon[lonind], llat[latind], "): ", xnow[iid, 0, 3])
            

            datanow = fh.variables['v'][ttind, 0, :, :]
            datanow, latind, lonind = nearest(datanow, llat, llon, latind, lonind)
            xnow[iid, 0, 4] = datanow
            
            print('Vvel(', llon[lonind], llat[latind], "): ", xnow[iid, 0, 4])
            fh.close()
            # wfile from sshfile

            # chlor_a(lat, lon), lat, lon(-180, 180), DOY, daily, mg/m^3
            # day_of_year = (tnow - datetime.datetime(tnow.year, 1, 1)).days + 1
            chlfile = glob.glob('../SOCO2_phase2/CHL/monthly/L3m_%04d%02d*%04d%02d*_4_*'%(year,tnow.month,year,tnow.month ))[0]
            print('chlfile now: ', chlfile)
            fh = nc.Dataset( chlfile )
            llat = fh.variables['lat'][:]
            llon = fh.variables['lon'][:]
            latind = np.argmin(abs(llat - latnow))
            lonind = np.argmin(abs(llon - lonnow))
            datanow = fh.variables['CHL1_mean'][:]
            datanow = np.ma.filled(datanow.astype(float), np.nan)

            datanow, latind, lonind = nearest_CHL(datanow, llat, llon, latind, lonind)
            xnow[iid, 0, 6] = datanow
            print('CHL-a(', llon[lonind], llat[latind], "): ", xnow[iid, 0, 6])
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
            xnow[iid, 0, 7] = datanow

            datanow = fh.variables['u10'][:]
            datanow = datanow[ttind, latind, lonind ]
            xnow[iid, 0, 8] = datanow

            datanow = fh.variables['v10'][:]
            datanow = datanow[ttind, latind, lonind ]
            xnow[iid, 0, 9] = datanow
            fh.close()

            print('sst( ', llon[lonind1], llat[latind1], "): ", xnow[iid, 0, 7])
            print('u10m(', llon[lonind], llat[latind], "): ", xnow[iid, 0, 8])

            print('v10m(', llon[lonind], llat[latind], "): ", xnow[iid, 0, 9])

            print('=======================================================')

    # 2016-1024-000000007

    ynow = np.load('bottle_Y_raw/%04d_smoothed.npy'%yy)
    print(xnow.shape, ynow.shape)
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!! NOW saving X: %04d_predictors.npy'%yy)
    np.save('bottle_X_raw/%04d_predictors.npy'%yy, xnow)