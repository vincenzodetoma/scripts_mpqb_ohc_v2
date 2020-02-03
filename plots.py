import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr

data_path = '/DataArchive/cyang/CMEMS_GLO_RAN/OHC_300_700_2000_BTM_SM_CELSIUS/Results/'
depths=['300','700','2000', 'btm']
what='trend'
var='_thetao'
models=['_cglo', '_foam', '_glor', '_oras', 'ensmean']
# Load the data
ds = xr.open_mfdataset(data_path+what+'*.nc')
#to_plot = xr.concat([xr.concat([ds[what+var+depths[d]+m] for m in models], 'model') for d in range(len(depths))], dim='depth')
to_plot = xr.concat([ds[what+var+depth+m] for depth in depths for m in models], 'model')


# This is the map projection we want to plot *onto*
map_proj = ccrs.Robinson()

p = to_plot.plot(
    transform=ccrs.PlateCarree(),  # the data's projection
    col="model",
    col_wrap=5,  # multiplot settings
    aspect=ds.dims["longitude"] / ds.dims["latitude"],  # for a sensible figsize
    subplot_kws={"projection": map_proj},  # the plot's projection
    extend='both',
    vmin=-100.,
    vmax=100.,
    cmap='seismic',
    cbar_kwargs={'label':'[(°C*m)/year]', 'shrink': 0.5}
)
l=0
# We have to set the map's options on all four axes
for ax in p.axes.flat:
    d = int(l/5)
    m = l % 5
    ax.coastlines()
    print(depths[d]+models[m])
    ax.set_title(what+var+depths[d]+models[m])
    # Without this aspect attributes the maps will look chaotic and the
    
    ax.set_aspect("equal") #"extent" attribute above will be ignored
    l=l+1

plt.show()
