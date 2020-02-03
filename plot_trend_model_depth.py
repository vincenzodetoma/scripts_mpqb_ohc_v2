import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import sys
plt.rcParams.update({'font.size' : 18})

model='_oras'
depth=sys.argv[1]

data_path='/DataArchive/cyang/CMEMS_GLO_RAN/OHC_300_700_2000_BTM_SM_CELSIUS'
trend_path=data_path+'/Results/'
varname='trend_thetao'+depth+model
#filename='trend_thetao'+depth+'_'+model+'1993-2017.nc'
filename='trend_thetao'+depth+model+'1993-2017.nc'
Period='1993-2017'

ds=xr.open_dataset(trend_path+filename)
trendvar = ds[varname]
fig = plt.figure(1, figsize=(12,8))
ax = plt.axes(projection=ccrs.Robinson())
ax.coastlines()
ax.gridlines()
colors = 'seismic'
trendvar.plot(ax=ax,
              transform=ccrs.PlateCarree(),
              extend='both',
              cmap=colors,
              cbar_kwargs={'label':'[(°C*m)/year]', 'shrink': 0.5},
              vmax=30.)
ax.set_title(trendvar.name+' '+Period)
fig.tight_layout()
fig.savefig(data_path+'/Figures/'+'trend_'+varname+depth+model+Period+'_extremes.png', dpi=300, transparent=True)
plt.show()
