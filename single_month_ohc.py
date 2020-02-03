#created by Vincenzo de Toma on Thu, 24 Oct 2019
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
#create paths where take and store the data
weight_path = #path where the vertical weights are stored
data_path = #path where there are 3d temperature data
out_path = #path were put the results of the calculation
#parse from the command line year and month
y=str(sys.argv[1])
m=str(sys.argv[2])
#loading the netcdf files for data and weights
dm = xr.open_dataset(weight_path+'/'+'mesh_mask.nc').rename({'t': 'time', 'z': 'depth'})
ds = xr.open_dataset(data_path+'/'+y+'/grepv2_monthly_'+y+m+'.nc')
#load the weight on the vertical level
dz = dm['e3t_1d'].squeeze()
#Find indices 
idx300 = np.where(ds.depth <= 300.)
idx700 = np.where(ds.depth <= 700.)
idx2000 = np.where(ds.depth <= 2000.)
idxbtm = np.where(ds.depth <= 6000.)
models=['_cglo', '_foam', '_glor', '_oras']
thetao_int = []
for lev in (300., 700., 2000., 6000.):
  dz2 = dz
  if(lev>ds.depth[len(ds.depth)-1].values):
    nlt = len(ds.depth)
  else:
    zt = 0
    for l in range(len(ds.depth)):
      zt = zt + dz[l]
      if(zt <lev):
        nlt = l
        ztf = zt
    nlt = nlt +1
    dz2[nlt] = lev - ztf

dz = dm['e3t_1d'].squeeze()
      
for i in models:

  T = ds['thetao'+i]
  thetao300 = (T*dz2).isel(depth=slice(idx300[0][0],idx300[0][-1] + 2)).sum(dim='depth').rename('thetao300'+i)
  thetao300.attrs['short_name'] = 'thetao300'+i
  thetao300.attrs['standard_name'] = 'temperature vertical integral top 300m from'+i
  thetao300.attrs['valid_min'] = thetao300.values.min()
  thetao300.attrs['valid_max'] = thetao300.values.max()
  thetao300.attrs['units'] = '°C*m'
  thetao_int.append(thetao300)
  thetao700 = (T*dz2).isel(depth=slice(idx700[0][0],idx700[0][-1] + 1)).sum(dim='depth').rename('thetao700'+i)
  thetao700.attrs['short_name'] = 'thetao700'+i
  thetao700.attrs['standard_name'] = 'temperature vertical integral top 700m from'+i
  thetao700.attrs['valid_min'] = thetao700.values.min()
  thetao700.attrs['valid_max'] = thetao700.values.max()
  thetao700.attrs['units'] = '°C*m'
  thetao_int.append(thetao700)
  thetao2000 = (T*dz2).isel(depth=slice(idx2000[0][0],idx2000[0][-1]+2)).sum(dim='depth').rename('thetao2000'+i)
  thetao2000.attrs['short_name'] = 'thetao2000'+i
  thetao2000.attrs['standard_name'] = 'temperature vertical integral top 2000m from'+i
  thetao2000.attrs['valid_min'] = thetao2000.values.min()
  thetao2000.attrs['valid_max'] = thetao2000.values.max()
  thetao2000.attrs['units'] = '°C*m' 
  thetao_int.append(thetao2000)
  thetaobtm = (T*dz2).sum(dim='depth').rename('thetaobtm'+i)
  thetaobtm.attrs['short_name'] = 'thetaobtm'+i
  thetaobtm.attrs['standard_name'] = 'temperature vertical integral top to bottom from'+i
  thetaobtm.attrs['valid_min'] = thetaobtm.values.min()
  thetaobtm.attrs['valid_max'] = thetaobtm.values.max()
  thetaobtm.attrs['units'] = '°C*m'
  thetao_int.append(thetaobtm)


dict_vars = {thetao_int[i].name : (thetao_int[i].dims, thetao_int[i].values, thetao_int[i].attrs) for i in range(len(thetao_int))}
out_ds = xr.Dataset(dict_vars, coords=thetao_int[0].coords)
out_ds.attrs['notes'] = 'Created from Vincenzo de Toma on 20/10/2019'
out_ds.to_netcdf(out_path+'/'+y+'/thetao_int_grepv2_monthly_'+y+m+'.nc')
dm.close()
ds.close()
out_ds.close()
