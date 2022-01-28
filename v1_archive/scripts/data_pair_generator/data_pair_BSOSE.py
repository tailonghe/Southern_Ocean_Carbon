import netCDF4 as nc
import numpy as np
import datetime
import gsw


era_t0 = datetime.datetime(year=1900, month=1, day=1)    
bsose_t0 = datetime.datetime(year=2008, month=1, day=1)   

era_dt = datetime.timedelta(hours=1)
bsose_dt = datetime.timedelta(seconds=1)
dt_3d = datetime.timedelta(days=3)

fh = nc.Dataset('iter105/bsose_i105_2008to2012_3day_DIC.nc')

bsose_times = fh.variables['time'][:]

bsose_times = bsose_times.astype(float)

bsose_times = np.array([datetime.timedelta(seconds=s) for s in bsose_times])

bsose_times = bsose_t0 + bsose_times

fh.close()

bsose_tstart = bsose_times - dt_3d
bsose_tend = bsose_times - datetime.timedelta(seconds=1)



_2dfields = [ 'SSH', 'pCO2', 'surfTflx']
_2dnames = [ 'ETAN', 'BLGPCO2', 'TFLUX']


_3dfields = [ 'Uvel', 'Vvel', 'Wvel', 'Chl', 'Alk', 'NO3', 'O2', 'PO4', 'Salt']
_3dnames = [ 'UVEL', 'VVEL', 'WVEL', 'BLGCHL', 'TRAC02', 'TRAC04', 'TRAC03', 'TRAC05', 'SALT']



xfiles = open("X_v9_list.txt", "w")
yfiles = open("Y_v9_list.txt", "w")


for iii in range(609):

    print(iii)
    xnow = np.zeros((317520, 1, 17))
    ynow = np.zeros((317520, 1, 52))


    for s in _2dfields:
        fh = nc.Dataset('iter105/bsose_i105_2008to2012_3day_' + s + '.nc')
        datanow = fh.variables[ _2dnames[ _2dfields.index(s)]][iii]
        datanow = np.reshape(datanow, (317520))

        xnow[ : , 0, _2dfields.index(s)] = datanow
        fh.close()


    for s in _3dfields:
        print(s)
        fh = nc.Dataset('iter105/bsose_i105_2008to2012_3day_' + s + '.nc')
        datanow = fh.variables[ _3dnames[ _3dfields.index(s)]][iii, 0, :, :]

        datanow = np.reshape(datanow, (317520))
        xnow[ : , 0, len(_2dfields) + _3dfields.index(s) ] = datanow

        fh.close()


    fh = nc.Dataset('iter105/bsose_i105_2008to2012_3day_Alk.nc')
    datanow = fh.variables[ 'Depth' ][:]
    datanow = (1000.0 - datanow) / 1000.0
    datanow = np.reshape(datanow, (317520))
    xnow[ : , 0, 12] = datanow
    fh.close()


    startnow = bsose_tstart[iii]
    endnow = bsose_tend[iii]

    print(startnow, endnow)

    startfile = '../ERA5/%04d/sea_surface_PTUV_%04d_%02d.third_remap.nc' % (startnow.year, startnow.year, startnow.month)
    endfile = '../ERA5/%04d/sea_surface_PTUV_%04d_%02d.third_remap.nc' % (endnow.year, endnow.year, endnow.month)


    if startfile == endfile:

        fh = nc.Dataset(startfile)

        era_times = fh.variables['time'][:]
        era_times = np.array([datetime.timedelta(hours=s) for s in era_times])
        era_times = era_times + era_t0

        indices = np.where( np.logical_and(era_times >= startnow, era_times < endnow) )

        sst = fh.variables['sst'][indices]
        ssp = fh.variables['sp'][indices]
        ssu = fh.variables['u10'][indices]
        ssv = fh.variables['v10'][indices]

        fh.close()
    else:

        fh = nc.Dataset(startfile)

        era_times = fh.variables['time'][:]
        era_times = np.array([datetime.timedelta(hours=s) for s in era_times])
        era_times = era_times + era_t0

        indices = np.where( np.logical_and(era_times >= startnow, era_times < endnow) )

        sst = fh.variables['sst'][indices]
        ssp = fh.variables['sp'][indices]
        ssu = fh.variables['u10'][indices]
        ssv = fh.variables['v10'][indices]

        fh.close()

        fh = nc.Dataset(endfile)

        era_times = fh.variables['time'][:]
        era_times = np.array([datetime.timedelta(hours=s) for s in era_times])
        era_times = era_times + era_t0

        indices = np.where( np.logical_and(era_times >= startnow, era_times < endnow) )

        if indices[0].shape[0] != 0:
            sst = np.concatenate( (sst, fh.variables['sst'][indices]), axis=0 )
            ssp = np.concatenate( (ssp, fh.variables['sp'][indices]), axis=0 )
            ssu = np.concatenate( (ssu, fh.variables['u10'][indices]), axis=0 )
            ssv = np.concatenate( (ssv, fh.variables['v10'][indices]), axis=0 )

        fh.close()

    print('ERA5 time slice shape (should be 72): ', sst.shape[0])

    sst = np.mean(sst, axis=0)
    ssp = np.mean(ssp * 1e-5, axis=0)    # convert to 1000 hPa
    ssu = np.mean(ssu, axis=0)
    ssv = np.mean(ssv, axis=0)

    sst = np.reshape(sst, (317520))
    ssp = np.reshape(ssp, (317520))
    ssu = np.reshape(ssu, (317520))
    ssv = np.reshape(ssv, (317520))

    xnow[ : , 0, 14 ] = sst
    xnow[ : , 0, 15 ] = ssu
    xnow[ : , 0, 16 ] = ssv

    
    fh = nc.Dataset('iter105/bsose_i105_2008to2012_3day_Salt.nc')
    salt = fh.variables[ 'SALT' ][iii, :, :, :]
    fh.close()

    fh = nc.Dataset('iter105/bsose_i105_2008to2012_3day_Theta.nc')
    theta = fh.variables[ 'THETA' ][iii, :, :, :]
    fh.close()

    ct = gsw.conversions.CT_from_pt(salt, theta)
    rhoa = gsw.density.sigma0(salt, ct)
    rho = rhoa + 1000.0                             
    rho = np.reshape(rho, (52, 317520, 1))        # density in kg/m^3

    fh = nc.Dataset('iter105/bsose_i105_2008to2012_3day_DIC.nc')
    datanow = fh.variables[ 'TRAC01' ][iii, :, :, :]
    datanow = np.reshape(datanow, (52, 317520, 1))

    datanow = datanow*1e6/rho # convert to umol/kg

    ynow[:, :, :] = np.moveaxis(datanow, 0, -1)

    print(np.nanmean(rho), np.nanmean(ynow))
    fh.close()


    np.save('X_v9/X_%03d.npy'%(iii), xnow)
    np.save('Y_v9/Y_%03d.npy'%(iii), ynow)


    xfiles.write('X_v9/X_%03d.npy'%(iii))
    xfiles.write('\n')

    yfiles.write('Y_v9/Y_%03d.npy'%(iii))
    yfiles.write('\n')
    


xfiles.close()
yfiles.close()



