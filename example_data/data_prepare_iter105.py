import numpy as np
import netCDF4 as nc
import glob
import datetime
import gsw



fh = nc.Dataset('iter105/bsose_i105_2008to2012_3day_DIC_1x1.nc')
llats = fh.variables['lat'][:]
llons = fh.variables['lon'][:]
llonb = list((llons[1:] + llons[:-1])/2)
llatb = list((llats[1:] + llats[:-1])/2)
llonb = np.array( [llonb[0]-1] + llonb + [llonb[-1]+1] )
llatb = np.array( [llatb[0]+1] + llatb + [llatb[-1]-1] )
times = fh.variables['time'][:]
fh.close()

# print(datetime.datetime(2012, 12, 1, 0, 0, 0) + datetime.timedelta(seconds=1)*2678400)
# print(datetime.datetime(2012, 12, 1, 0, 0, 0) + datetime.timedelta(seconds=1)*3024000)
# print(datetime.datetime(2012, 12, 1, 0, 0, 0) + datetime.timedelta(seconds=1)*191808000)

etimes = datetime.datetime(2008, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=1)*times
stimes = etimes - datetime.timedelta(days=3)
print('start dates: ', stimes[0], etimes[0])
print('end dates: ', stimes[-1], etimes[-1])

def to_ERAfile(dt):
	return 'ERA5/%04d/ERA5_Met_%04d_%d_1x1.nc' % (dt.year, dt.year, dt.month)

def to_Tref(dt):
	dt = dt - datetime.datetime(1900, 1, 1, 0, 0, 0)
	return dt.total_seconds()/3600


for i in range(609):
# for i in range(5):

	fh = nc.Dataset('iter105/bsose_i105_2008to2012_3day_DIC_1x1.nc')
	dic = fh.variables['TRAC01'][i, :48, :, :]
	fh.close()
	fh = nc.Dataset('iter105/bsose_i105_2008to2012_3day_Salt_1x1.nc')
	salt = fh.variables[ 'SALT' ][i, :48, :, :]
	fh.close()
	fh = nc.Dataset('iter105/bsose_i105_2008to2012_3day_Theta_1x1.nc')
	theta = fh.variables[ 'THETA' ][i, :48, :, :]
	fh.close()
	ct = gsw.conversions.CT_from_pt(salt, theta)
	rhoa = gsw.density.sigma0(salt, ct)
	rho = rhoa + 1000.0 # density in kg/m^3
	dic = dic*1e6/rho   # convert to umol/kg


	fh = nc.Dataset('iter105/bsose_i105_2008to2012_3day_Chl_1x1.nc')
	chl = fh.variables['BLGCHL'][i, 0, :, :]
	fh.close()

	fh = nc.Dataset('iter105/bsose_i105_2008to2012_3day_SSH_1x1.nc')
	ssha = fh.variables['ETAN'][i, :, :]
	fh.close()

	fh = nc.Dataset('iter105/bsose_i105_2008to2012_3day_pCO2_1x1.nc')
	pco2 = fh.variables['BLGPCO2'][i, :, :]
	fh.close()

	fh = nc.Dataset('iter105/bsose_i105_2008to2012_3day_surfTflx_1x1.nc')
	tflux = fh.variables['TFLUX'][i, :, :]
	fh.close()


	fh = nc.Dataset('iter105/bsose_i105_2008to2012_3day_Uvel_1x1.nc')
	uvel = fh.variables['UVEL'][i, 0, :, :]
	fh.close()

	fh = nc.Dataset('iter105/bsose_i105_2008to2012_3day_Vvel_1x1.nc')
	vvel = fh.variables['VVEL'][i, 0, :, :]
	fh.close()

	fh = nc.Dataset('iter105/bsose_i105_2008to2012_3day_Wvel_1x1.nc')
	wvel = fh.variables['WVEL'][i, 0, :, :]
	fh.close()

	stimenow = stimes[i]
	etimenow = etimes[i]

	print(stimenow, etimenow)
	sfile = to_ERAfile(stimenow)
	efile = to_ERAfile(etimenow)
	stref = to_Tref(stimenow)
	etref = to_Tref(etimenow)

	print(sfile, efile)
	if sfile != efile:
		fh = nc.Dataset(sfile)
		eratimes1 = fh.variables['time'][:]
		u10m1 = fh.variables['u10'][:, :, :]
		v10m1 = fh.variables['v10'][:, :, :]
		sst1 = fh.variables['sst'][:, :, :]
		eralons = fh.variables['lon'][:]
		fh.close()

		fh = nc.Dataset(efile)
		eratimes2 = fh.variables['time'][:]
		u10m2 = fh.variables['u10'][:, :, :]
		v10m2 = fh.variables['v10'][:, :, :]
		sst2 = fh.variables['sst'][:, :, :]
		fh.close()

		eratimes = np.concatenate([eratimes1, eratimes2], axis=0)
		u10m = np.concatenate([u10m1, u10m2], axis=0)
		v10m = np.concatenate([v10m1, v10m2], axis=0)
		sst = np.concatenate([sst1, sst2], axis=0)
		mask = np.where( np.logical_and(eratimes >= stref, eratimes < etref) )[0]
		u10m = u10m[mask, :, :]
		v10m = v10m[mask, :, :]
		sst = sst[mask, :, :]

	else:
		fh = nc.Dataset(sfile)
		eralons = fh.variables['lon'][:]
		eratimes = fh.variables['time'][:]
		mask = np.where( np.logical_and(eratimes >= stref, eratimes < etref) )[0]
		u10m = fh.variables['u10'][mask, :, :]
		v10m = fh.variables['v10'][mask, :, :]
		sst = fh.variables['sst'][mask, :, :]
		fh.close()

	print(u10m.shape, v10m.shape, sst.shape)

	u10m = np.mean(u10m, axis=0)
	v10m = np.mean(v10m, axis=0)
	sst = np.mean(sst, axis=0)

	# change to 0~360
	eralons = np.roll(eralons, 180)
	u10m = np.roll(u10m, 180, axis=-1)
	v10m = np.roll(v10m, 180, axis=-1)
	sst = np.roll(sst, 180, axis=-1)

	# pco2: umol/m2/s
	# ssha: m
	# tflx: W/m^2
	# uvel: m/s
	# vvel: m/s
	# wvel: um/s
	# chl: mg/m3
	# u10m: m/s
	# v10m: m/s
	# sst: C
	# shape: 10, 56, 360
	xnow = np.stack([pco2*1e6, ssha, tflux, uvel, vvel, wvel*1e6, chl, u10m, v10m, sst-273.15])
	# cutting off boundary
	xnow = xnow[:, 8:, :]

	# DIC: umol/kg
	ynow = dic[:, 8:, :]
	print(i, llats[8:].shape, llatb[8:].shape)


	np.savez('X/X_%03d.npz'%(i), lat=llats[8:], lon=llons, latb=llatb[8:], lonb=llonb, x=xnow)
	np.savez('Y/Y_%03d.npz'%(i), lat=llats[8:], lon=llons, latb=llatb[8:], lonb=llonb, y=ynow)