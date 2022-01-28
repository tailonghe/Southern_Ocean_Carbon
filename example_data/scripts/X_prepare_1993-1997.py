import numpy as np
import netCDF4 as nc
import glob
import datetime
import gsw


eralons = np.arange(0.5, 360.5, 1)
eralats = np.arange(-32.5, -80.5, -1)

# def to_ERAfile(dt):
# 	return 'ERA5/%04d/ERA5_Met_%04d_%d_1x1.nc' % (dt.year, dt.year, dt.month)

# def to_Tref(dt):
# 	dt = dt - datetime.datetime(1900, 1, 1, 0, 0, 0)
# 	return dt.total_seconds()/3600



for yy in range(1993, 1998):
	base = datetime.datetime(yy, 1, 1, 0, 1, 0)
	stimes = np.array([base + datetime.timedelta(days=x) for x in range(0, 365, 5)])
	etimes = stimes + datetime.timedelta(days=5)


	# predictors = np.zeros((73, 48, 360, 10))

	for doyidx in range(73):

		stimenow = stimes[doyidx]
		timenow = stimenow
		etimenow = etimes[doyidx]
		predictors = np.zeros((1, 48, 360, 10))
		temp = np.zeros((5, 48, 360, 10))
		
		print('Point check: ', stimenow, etimenow)
		curr = 0
		while timenow < etimenow:
			# pco2 in mu atm
			fh = nc.Dataset('spco2_MPI-SOM_FFN_v2020_1x1.nc')
			_times = fh.variables['time'][:]
			_times = datetime.datetime(2000, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=1)*_times
			_tidx = np.argmin( np.abs( timenow - _times) )
			pco2 = fh.variables['spco2_smoothed'][_tidx, 8:, :]
			# print(np.roll(fh.variables['lon'][:], 180, axis=-1))
			fh.close()
			pco2 = np.roll(pco2, 180, axis=-1)
			print('pCO2 check: ')
			print(timenow, _times[_tidx])
			pco2[ pco2 > 1.e+18 ] = np.nan

			# ssha in m
			file = glob.glob('CMEMS_SSHA/%04d/%02d/dt_global_allsat_phy_l4_%04d%02d%02d_*.nc_1x1'%
				(timenow.year, timenow.month, timenow.year, timenow.month, timenow.day))[0]
			print('SSHA file: ', file)
			fh = nc.Dataset(file)
			ssha = fh.variables['sla'][0, 8:, :]
			fh.close()
			ssha[ssha < -2147480000] = np.nan
			# tflux in W/m^2
			# u10m, 
			# v10m, 
			# sst-273.15
			fh = nc.Dataset('../Phase1_predictors/ERA5/%04d/ERA5_Met_%04d_%d_1x1.nc' % (timenow.year, timenow.year, timenow.month))
			_tidx = timenow.day - 1
			tflx = fh.variables['slhf'][:] + fh.variables['sshf'][:]  + fh.variables['ssr'][:]  + fh.variables['str'][:]
			tflx = np.nanmean(tflx[ _tidx*24:(_tidx+1)*24, 8:, :], axis=0)/3600.  # convert J m^-2 to W m^-2
			u10m = np.nanmean(fh.variables['u10'][ _tidx*24:(_tidx+1)*24, 8:, :], axis=0)
			v10m = np.nanmean(fh.variables['v10'][ _tidx*24:(_tidx+1)*24, 8:, :], axis=0)
			sst = np.nanmean(fh.variables['sst'][ _tidx*24:(_tidx+1)*24, 8:, :], axis=0) - 273.15 # convert to celsius
			# print(np.roll(fh.variables['lon'][:], 180, axis=-1))
			fh.close()
			tflx = np.roll(tflx, 180, axis=-1)
			u10m = np.roll(u10m, 180, axis=-1)
			v10m = np.roll(v10m, 180, axis=-1)
			sst = np.roll(sst, 180, axis=-1)

			# uvel in m/s 
			print('UV check: ')
			fh = nc.Dataset('OSCAR_UV/oscar_vel%04d_1x1.nc' % (timenow.year))
			_times = fh.variables['time'][:]
			_times = (_times[1:] + _times[:-1])/2
			_times = np.array(list(_times) + [_times[-1]+5])
			_times = datetime.datetime(1992, 10, 5, 0, 0, 0) + datetime.timedelta(days=1)*_times
			_tidx = np.argmin( np.abs( timenow - _times) )
			uvel = fh.variables['u'][_tidx, 0, 8:, :]
			vvel = fh.variables['v'][_tidx, 0, 8:, :]
			# print(np.roll(fh.variables['lon'][:], 20, axis=-1))
			print(_times[_tidx])
			fh.close()
			uvel = np.roll(uvel, 20, axis=-1)
			vvel = np.roll(vvel, 20, axis=-1)
			print(timenow, _times[_tidx])

			# w in um/s, skipped...

			# chl in ug/m3
			file = glob.glob('CHL/monthly/L3m_%04d%02d01*.nc_1x1' %
				   (timenow.year, timenow.month) )
			inc = 1
			while len(file) == 0:
				file = glob.glob('CHL/monthly/L3m_%04d%02d01*.nc_1x1' %
					   (timenow.year+inc, timenow.month) )
				inc += 1

			if len(file) == 0:
				print('NO CHL file found!')
				chl = np.zeros((48, 360))
				chl[:, :] = np.nan
			else:
				file = file[0]
				print('CHL file: ', file)
				fh = nc.Dataset(file)
				chl = fh.variables['CHL1_mean'][8:, :] 
				# print(np.roll(fh.variables['lon'][:], 180, axis=-1))
				fh.close()
				chl = np.roll(chl, 180, axis=-1)
				chl[chl < 0] = np.nan

			temp[curr, :, :, 0] = pco2
			temp[curr, :, :, 1] = ssha
			temp[curr, :, :, 2] = tflx
			temp[curr, :, :, 3] = uvel
			temp[curr, :, :, 4] = vvel
			temp[curr, :, :, 5] = ssha   # set W vel as sea level height for now..
			temp[curr, :, :, 6] = chl
			temp[curr, :, :, 7] = u10m
			temp[curr, :, :, 8] = v10m
			temp[curr, :, :, 9] = sst

			timenow = timenow + datetime.timedelta(days=1)
			curr += 1
		
		temp[temp == 0] = np.nan
		predictors[0] = np.nanmean(temp, axis=0)
		predictors[0, :, :, 5] = (temp[-1, :, :, 5] - temp[0, :, :, 5])/432000.*1e6 # W in um/s
		
		np.savez('X/X_%04d_%02d.npz' % (yy, doyidx), x=predictors)

