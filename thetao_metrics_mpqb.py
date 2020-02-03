'''

   Created by Vincenzo de Toma on 4 Jan 2020

'''
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from scipy.stats import pearsonr
from statsmodels.tsa.seasonal import seasonal_decompose
from mk_test import mk_test
from trend_2d_parallel import trend_2d_parallel
from statsmodels.tsa.seasonal import seasonal_decompose
import sys
plt.rcParams.update({'font.size' : 18})
#load the datasets 
data_path = '/DataArchive/cyang/CMEMS_GLO_RAN/OHC_300_700_2000_BTM_SM_CELSIUS/'
filename = 'thetao_int_grepv2_monthly_1993_2017.nc'
dataset = xr.open_dataset(data_path+filename)
varname='thetao'
depth=str(sys.argv[1])
models=['_cglo', '_foam', '_glor', '_oras']
fig=[0]
Nobs=len(models)
lat = dataset.latitude
lon = dataset.longitude
time=dataset.time
Nlat=len(lat)
Nlon=len(lon)
Period='1993-2017'
units='[°C*m]'
i=1
#take the different depths contribution and models
single_products = [dataset[varname+depth+m] for m in models]
dataset.close()
#calculate the reference products: ensemble mean at the given depth
reference_product = sum(single_products[m] for m in range(Nobs)) / Nobs
reference_product = reference_product.rename('ensmean_'+varname+depth)
#calculate yearly means
annual_means = [single_products[m].groupby(time.dt.year).mean(dim='time') for m in range(Nobs)]
#yearly means ensemble mean
annual_reference = sum(annual_means[m].rename('ens_ann_mean'+varname+depth) for m in range(Nobs)) / Nobs
#calculating the climatology for all the members and for the ensemble mean
single_climatologies = [annual_means[m].mean(dim='year').rename('clim_'+varname+depth+models[m]) for m in range(Nobs)] #SUGGESTION: sum instead of the mean to have W/m^2?
#calculate the reference climatology and use as bounds for colormap of each
ref_climatology = annual_reference.mean(dim='year').rename('ensmean_clim_'+varname+depth)
#plot the climatologies for each member
for m in range(Nobs):
  fig.append(plt.figure(i, figsize=(12,8)))
  ax = plt.axes(projection=ccrs.Robinson())
  ax.coastlines()
  ax.gridlines()
  colors = 'nipy_spectral'
  single_climatologies[m].plot(ax=ax, 
                               transform=ccrs.PlateCarree(),
                               extend='both',
                               cmap=colors, 
                               cbar_kwargs={'label':units, 'shrink': 0.5}, 
                               vmin=-500., vmax=ref_climatology.max())
  ax.set_title(single_climatologies[m].name+' '+Period)
  fig[i].tight_layout()
  single_climatologies[m].to_netcdf(data_path+'/Results/'+'clim_'+varname+depth+models[m]+Period+'.nc')
  fig[i].savefig(data_path+'/Figures/'+'clim_'+varname+depth+models[m]+Period+'.png', dpi=300, transparent=True)
  #fig[i].show()
  i=i+1

print('\n Finished climatologies \n')

plt.show()

#plot the climatology for the reference dataset
fig.append(plt.figure(i, figsize=(12,8)))
ax = plt.axes(projection=ccrs.Robinson())
ax.coastlines()
ax.gridlines()
colors = 'nipy_spectral'
ref_climatology.plot(ax=ax, 
                     transform=ccrs.PlateCarree(), 
                     extend='both', 
                     cmap=colors, 
                     cbar_kwargs={'label':units,'shrink': 0.5},
                     vmin=-500., vmax = ref_climatology.max())
ax.set_title(ref_climatology.name+' '+Period)
fig[i].tight_layout()
ref_climatology.to_netcdf(data_path+'/Results/'+'clim_'+varname+depth+'ensmean'+Period+'.nc')
fig[i].savefig(data_path+'/Figures/'+'clim_'+varname+depth+'ensmean'+Period+'.png', dpi=300, transparent=True)
#fig[i].show()
i=i+1

print('\n Finished ensemble mean climatology \n')

#climatology biases with respect to the ref
biases_climatologies = [(single_climatologies[m] - ref_climatology).rename('bias_'+varname+depth+models[m]) for m in range(Nobs)]
#plot the biases for each member
for m in range(Nobs):
  fig.append(plt.figure(i, figsize=(12,8)))
  ax = plt.axes(projection=ccrs.Robinson())
  ax.coastlines()
  ax.gridlines()
  colors = 'seismic'
  biases_climatologies[m].plot(ax=ax,
                               transform=ccrs.PlateCarree(),
                               extend='both',
                               cmap=colors,
                               cbar_kwargs={'label':units, 'shrink': 0.5}, 
                               vmin=-750., vmax=750.)
  ax.set_title(biases_climatologies[m].name+' '+Period)
  fig[i].tight_layout()
  biases_climatologies[m].to_netcdf(data_path+'/Results/'+'bias_'+varname+depth+models[m]+Period+'.nc')
  fig[i].savefig(data_path+'/Figures/'+'bias_'+varname+depth+models[m]+Period+'.png', dpi=300, transparent=True)
  #fig[i].show()
  i=i+1

print('\n Finished biases of each member \n')

#climatology for the ensemble spread
spread_climatology = sum((biases_climatologies[m])**2. for m in range(Nobs)) / (Nobs*(Nobs-1))
spread_climatology  = spread_climatology.rename('en_spread_clim_'+varname+depth)
#plot the ensemble spread climatology
fig.append(plt.figure(i, figsize=(12,8)))
ax = plt.axes(projection=ccrs.Robinson())
ax.coastlines()
ax.gridlines()
colors = 'Greens'
spread_climatology.plot(ax=ax, 
                        transform=ccrs.PlateCarree(), 
                        extend='both', 
                        cmap=colors, 
                        cbar_kwargs={'label':units,'shrink': 0.5},
                        vmin=0., vmax=2000.)
ax.set_title(spread_climatology.name+' '+Period)
fig[i].tight_layout()
spread_climatology.to_netcdf(data_path+'/Results/'+'clim_'+varname+depth+'ensspread'+Period+'.nc')
fig[i].savefig(data_path+'/Figures/'+'clim_'+varname+depth+'ensspread'+Period+'.png', dpi=300, transparent=True)
#fig[i].show()
i=i+1

print('\n Finished ensemble spread climatology \n')

#gridpoint RMSD
grid_rmsd = []
for m in range(Nobs):
  rmsd = np.sqrt(((annual_means[m] - annual_reference)**2).mean(dim='year'))
  grid_rmsd.append(rmsd.rename('rmsd_'+varname+depth+models[m]))

for m in range(Nobs):
  fig.append(plt.figure(i, figsize=(12,8)))
  ax = plt.axes(projection=ccrs.Robinson())
  ax.coastlines()
  ax.gridlines()
  colors = 'Oranges'
  grid_rmsd[m].plot(ax=ax,
                    transform=ccrs.PlateCarree(),
                    extend='both',
                    cmap=colors,
                    cbar_kwargs={'label':units, 'shrink': 0.5},
                    vmin=0., vmax=400.)
  ax.set_title(grid_rmsd[m].name+' '+Period)
  fig[i].tight_layout()
  grid_rmsd[m].to_netcdf(data_path+'/Results/'+'rmsd_'+varname+depth+models[m]+Period+'.nc')
  fig[i].savefig(data_path+'/Figures/'+'rmsd_'+varname+depth+models[m]+Period+'.png', dpi=300, transparent=True)
  #fig[i].show()
  i=i+1

print('\n Finished gridpoint rmsd \n')

del single_climatologies
del ref_climatology
del biases_climatologies
del spread_climatology

plt.show()

#trend analysis
frequency = 12*2 #12=months; 2=years
nsplit = 16
trend_maps = []
for m in range(Nobs):
    trend_matrix, trend_pvalue, trend_significance = trend_2d_parallel(single_products[m].fillna(0), single_products[m].fillna(0), lat, lon, time, single_products[m][0,0,0], frequency, nsplit)
    trend_matrix = xr.DataArray(trend_matrix,
                                coords={'latitude':lat, 'longitude':lon},
                                dims=['latitude','longitude'])
    trend_maps.append(trend_matrix.rename('trend_'+varname+depth+models[m]))

#same calculatin for the ensemble mean
trend_matrix, trend_pvalue, trend_significance = trend_2d_parallel(reference_product.fillna(0), reference_product.fillna(0), lat, lon, time, reference_product[0,0,0], frequency, nsplit)
trend_matrix = xr.DataArray(trend_matrix,
                            coords={'latitude':lat, 'longitude':lon},
                            dims=['latitude','longitude'])
trend_maps.append(trend_matrix.rename('trend_'+varname+depth+'ensmean'))
models.append('_ensmean')
#plot the trend maps
for m in range(Nobs+1):
  fig.append(plt.figure(i, figsize=(12,8)))
  ax = plt.axes(projection=ccrs.Robinson())
  ax.coastlines()
  ax.gridlines()
  colors = 'seismic'
  trend_maps[m].plot(ax=ax,
                     transform=ccrs.PlateCarree(),
                     extend='both',
                     cmap=colors,
                     cbar_kwargs={'label':'[(°C*m)/year]', 'shrink': 0.5},
                     vmin=-100., vmax=100.)
  ax.set_title(trend_maps[m].name+' '+Period)
  fig[i].tight_layout()
  trend_maps[m].to_netcdf(data_path+'/Results/'+'trend_'+varname+depth+models[m]+Period+'.nc')
  fig[i].savefig(data_path+'/Figures/'+'trend_'+varname+depth+models[m]+Period+'.png', dpi=300, transparent=True)
  #fig[i].show()
  i=i+1

print('\n Finished Trend Analysis \n')

rho=1026.
c_p=4000.
#Global mean time series
lat_to_km = 25.*10**3
lon_to_km = np.cos(lat*np.pi/180.)*25.*10**3
global_means = [(rho*c_p*lat_to_km*lon_to_km*single_products[m]).sum(dim=['latitude', 'longitude']).rename('glob_'+varname+depth+models[m]) for m in range(Nobs)]
global_means.append((rho*c_p*lat_to_km*lon_to_km*reference_product).sum(dim=['latitude', 'longitude']).rename('glob_'+varname+depth+'_ensmean'))
models.append('_ensmean')
fig.append(plt.figure(i, figsize=(12,8)))
ax = plt.axes()
for m in range(Nobs+1):
  global_means[m] = global_means[m] - global_means[m].groupby(time.dt.year).mean(dim='time')[0]
  global_means[m].to_netcdf(data_path+'/Results/'+'glob_'+varname+depth+models[m]+Period+'.nc')
  global_means[m].plot(ax=ax, label=global_means[m].name, marker='o')
  ax.set_title('Global time series '+varname+depth+' '+Period)
  ax.set_ylabel('Global OHC '+depth+' [Joules]')
  ax.legend(loc='best')
  fig[i].savefig(data_path+'/Figures/'+'glob_'+varname+depth+models[m]+Period+'.png', dpi=300, transparent=True)
  #i=i+1

print('\n Finished Global mean time series \n')
plt.show()

rho=1026.
c_p=4000.
#Global mean yearly time series
lat_to_km = 25.*10**3
lon_to_km = np.cos(lat*np.pi/180.)*25.*10**3
global_means_y = [(rho*c_p*lat_to_km*lon_to_km*single_products[m]).sum(dim=['latitude', 'longitude']).groupby(time.dt.year).mean(dim='time').rename('glob_'+varname+depth+models[m]) for m in range(Nobs)]
global_means_y.append((rho*c_p*lat_to_km*lon_to_km*reference_product).sum(dim=['latitude', 'longitude']).groupby(time.dt.year).mean(dim='time').rename('glob_'+varname+depth+'_ensmean'))
models.append('_ensmean')
fig.append(plt.figure(i, figsize=(12,8)))
ax = plt.axes()
for m in range(Nobs+1):
  global_means_y[m] = global_means_y[m] - global_means_y[m][0]
  global_means_y[m].to_netcdf(data_path+'/Results/'+'glob_'+varname+depth+models[m]+Period+'_y.nc')
  global_means_y[m].plot(ax=ax, label=global_means_y[m].name, marker='o')
  ax.set_title('Global yearly time series '+varname+depth+' '+Period)
  ax.set_ylabel('Global OHC '+depth+' [Joules]')
  ax.legend(loc='best')
  fig[i].savefig(data_path+'/Figures/'+'glob_'+varname+depth+models[m]+Period+'_y.png', dpi=300, transparent=True)
  #i=i+1

print('\n Finished Global mean time series \n')
plt.show()


r_pearsons = []
p_values = []
#New gridpoint pearson 
for m in range(Nobs):
  r = xr.DataArray(np.zeros((Nlat,Nlon)),
                   coords={'latitude':lat, 'longitude':lon},
                   dims=['latitude','longitude'])
  p = xr.DataArray(np.zeros((Nlat,Nlon)),
                   coords={'latitude':lat, 'longitude':lon},
                   dims=['latitude','longitude'])
  for ii in range(Nlat):
    for jj in range(Nlon):
      result_r = pearsonr(annual_means[m][:,ii,jj], annual_reference[:,ii,jj])[0] if ~np.isnan(annual_means[m][0,ii,jj]) else np.nan
      result_p = pearsonr(annual_means[m][:,ii,jj], annual_reference[:,ii,jj])[1] if ~np.isnan(annual_means[m][0,ii,jj]) else np.nan
      r[ii,jj] = result_r
      p[ii,jj] = result_p
  r_pearsons.append(r.rename('R_'+varname+depth+models[m]))
  p_values.append(p.rename('P_val_'+varname+depth+models[m]))
  r_pearsons[m].to_netcdf(data_path+'/Results/'+'R_'+varname+depth+models[m]+Period+'.nc')
  p_values[m].to_netcdf(data_path+'/Results/'+'P_'+varname+depth+models[m]+Period+'.nc')
  #plot the results
  fig.append(plt.figure(i, figsize=(12,8)))
  ax = plt.axes(projection=ccrs.Robinson())
  ax.coastlines()
  ax.gridlines()
  colors = 'bwr'
  r_pearsons[m].plot(ax=ax,
                     transform=ccrs.PlateCarree(),
                     extend='both',
                     cmap=colors,
                     cbar_kwargs={'label':' ', 'shrink': 0.5},
                     vmin=-1., vmax=+1.)
  ax.set_title(r_pearsons[m].name+' '+Period)
  fig[i].tight_layout()
  fig[i].savefig(data_path+'/Figures/'+'rpearson_'+varname+depth+models[m]+Period+'.png', dpi=300, transparent=True)
  #fig[i].show()
  i=i+1

print('\n Finished pearson correlation coefficients \n')


#Taylor plot

#add the annual global means?

#Other and optionals 
