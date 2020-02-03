import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from statsmodels.tsa.seasonal import seasonal_decompose
from mk_test import mk_test

data_path = '/DataArchive/cyang/CMEMS_GLO_RAN/OHC_300_700_2000_BTM_SM_CELSIUS/Results/'
depths=['300','700','2000', 'btm']
what='glob'
var='_thetao'
models=['_cglo', '_foam', '_glor', '_oras', '_ensmean']
# Load the data
ds = xr.open_mfdataset(data_path+what+'*.nc')
mon_plot = xr.concat([xr.concat([ds[what+var+depths[d]+m] for m in models], dim='model') for d in range(len(depths))], dim='depth')
mon_plot.coords['model'] = [m.strip('_') for m in models]
mon_plot.coords['depth'] = [d for d in depths]
#to_plot = xr.concat([ds[what+var+depth+m] for depth in depths for m in models], 'model')
month_anom = mon_plot
p = month_anom.isel(model=slice(0,5)).plot(
    hue='model',
    col='depth',
    col_wrap=1,
    marker='o',
    markersize=2,
    figsize=(8,10)# multiplot settings
)

l=0

for ax in p.axes.flat:
    d = l%4
    ax.set_title(what+' OHC '+depths[d])
    ax.set_ylabel('Joules')
    ax.figure.tight_layout()
    ax.set_aspect('auto')
    l=l+1

plt.show()



print(mon_plot)
year_plot = mon_plot.groupby(ds.time.dt.year).mean('time')
year_anom = year_plot - year_plot.isel(year=0)

p = year_anom.isel(model=slice(0,5)).plot(
    hue='model',
    col='depth',
    col_wrap=1,
    marker='o',
    markersize=2, 
    figsize=(8,10)# multiplot settings
)

l=0
for ax in p.axes.flat:
    d = l%4
    ax.set_title(what+' OHC '+depths[d])
    ax.set_ylabel('Joules')
    ax.figure.tight_layout()
    ax.set_aspect('auto')
    l=l+1

plt.show()


year_start=1993
year_end=2017
n_years=25
n_months=300
frequency = int(12*2)
half_frequency = int(frequency/2)
indexes=[np.array(range(i*12,(i+1)*12)).astype(int) for i in range(np.int(n_years))]

for d in mon_plot.coords['depth'].values:
  for i in mon_plot.coords['model'].values:
    result = seasonal_decompose(mon_plot.sel(model=i, depth=d), model='additive', filt=None, freq=frequency, two_sided=True, extrapolate_trend=0)
    monthly_trend_component = result.trend[half_frequency:-half_frequency]
    yearly_trend_component=[np.average(monthly_trend_component[indexes[t]]) for t in range(int(n_years-frequency/12))]
    trend,h,p,z,slope, std_conf = mk_test(yearly_trend_component, np.linspace(1,n_years-frequency/12,n_years-frequency/12), True, 0.05)
    plt.plot(np.linspace(year_start,year_end,n_months), result.trend, label=i, linewidth=1)
    print(str(d)+str(i) + ' trend +- 2*std & test: ', slope, std_conf*2.0, trend)
    #print(str(d)+str(i) + ' trend +- 2*std & test: ',slope/(1.1414326*10**22), std_conf*2.0/(1.1414326*10**22), "W/m^2", trend)
  
  plt.title(what+' OHC '+d+' trend component')
  plt.xlabel('Years')
  plt.ylabel('J/year')
  plt.legend()

  #plt.savefig(figures_path + 'global_averaged_sst_trend.png',dpi=300)
  plt.show()

  print("trend +- 2*std & test: ", slope, std_conf*2.0, trend)


