#created by Vincenzo de Toma on Thu, 24 Oct 2019
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
#create paths where take and store the data
weight_path = '/store3/c3s_511/MODEL7/GRID/ORAS5'
data_path = '/DataArchive/cyang/CMEMS_GLO_RAN/GREP'
out_path='/DataArchive/cyang/CMEMS_GLO_RAN/tmp'
#parse from the command line year and month
y=str(sys.argv[1])
m=str(sys.argv[2])
#loading the netcdf files for data and weights
dm = xr.open_dataset(weight_path+'/'+'mesh_mask.nc').rename({'t': 'time', 'z': 'depth'})
ds = xr.open_dataset(data_path+'/'+y+'/grepv2_monthly_'+y+m+'.nc')
#load the weight on the vertical level
dz = dm['e3t_1d'].squeeze()
#define constants
c_p = 4000.
rho = 1020.
#Find indices 
idx300 = np.where(ds.depth <= 300.)
idx700 = np.where(ds.depth <= 700.)
idx2000 = np.where(ds.depth <= 2000.)
idxbtm = np.where(ds.depth <= 6000.)
models=['_cglo', '_foam', '_glor', '_oras']
ohc = []
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

  T = ds['thetao'+i] + 273.15 #conversion from celsius to kelvin
  sohtc300 = c_p*rho*(T*dz2).isel(depth=slice(idx300[0][0],idx300[0][-1] + 2)).sum(dim='depth').rename('sohtc300'+i)
  sohtc300.attrs['short_name'] = 'ohc300m'+i
  sohtc300.attrs['standard_name'] = 'ocean heat content top 300m from'+i
  sohtc300.attrs['valid_min'] = sohtc300.values.min()
  sohtc300.attrs['valid_max'] = sohtc300.values.max()
  sohtc300.attrs['units'] = 'J/m^2'
  ohc.append(sohtc300)
  sohtc700 = c_p*rho*(T*dz2).isel(depth=slice(idx700[0][0],idx700[0][-1] + 1)).sum(dim='depth').rename('sohtc700'+i)
  sohtc700.attrs['short_name'] = 'ohc700m'+i
  sohtc700.attrs['standard_name'] = 'ocean heat content top 700m from'+i
  sohtc700.attrs['valid_min'] = sohtc700.values.min()
  sohtc700.attrs['valid_max'] = sohtc700.values.max()
  sohtc700.attrs['units'] = 'J/m^2'
  ohc.append(sohtc700)
  sohtc2000 = c_p*rho*(T*dz2).isel(depth=slice(idx2000[0][0],idx2000[0][-1]+2)).sum(dim='depth').rename('sohtc2000'+i)
  sohtc2000.attrs['short_name'] = 'ohc2000m'+i
  sohtc2000.attrs['standard_name'] = 'ocean heat content top 2000m from'+i
  sohtc2000.attrs['valid_min'] = sohtc2000.values.min()
  sohtc2000.attrs['valid_max'] = sohtc2000.values.max()
  sohtc2000.attrs['units'] = 'J/m^2' 
  ohc.append(sohtc2000)
  sohtcbtm = c_p*rho*(T*dz2).sum(dim='depth').rename('sohtcbtm'+i)
  sohtcbtm.attrs['short_name'] = 'ohcbtmm'+i
  sohtcbtm.attrs['standard_name'] = 'ocean heat content top to bottom from'+i
  sohtcbtm.attrs['valid_min'] = sohtcbtm.values.min()
  sohtcbtm.attrs['valid_max'] = sohtcbtm.values.max()
  sohtcbtm.attrs['units'] = 'J/m^2'
  ohc.append(sohtcbtm)


dict_vars = {ohc[i].name : (ohc[i].dims, ohc[i].values, ohc[i].attrs) for i in range(len(ohc))}
out_ds = xr.Dataset(dict_vars, coords=ohc[0].coords)
out_ds.attrs['notes'] = 'Created from Vincenzo de Toma on 20/10/2019'
out_ds.to_netcdf(out_path+'/'+y+'/ohc_grepv2_monthly_'+y+m+'.nc')
dm.close()
ds.close()
out_ds.close()
