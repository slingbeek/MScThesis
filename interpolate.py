#!/usr/bin/python
"""This module interpolates the data from hybrid to isentropic levels.

This file will read the cesm data on hybrid levels and interpolate
them onto isentropic levels. The results are stored in separate files.
A basis for the interpolation procedure can be found in: "On the maintenance of 
potential vorticity in isentropic coordinates. Edouard et al. 
Q.J.R Meteorol. Soc. (1997), 123, pp 2069-2094".
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import sys
import glob
import numpy as np
from numba import float32, float64, guvectorize
from datetime import datetime


file, model, timing = input("Enter file name, model (cesm1/cesm2) and time interval (monthly/daily): ").split(",")
if model == "cesm1":
    if timing == "monthly":
        saveloc = "/home/slingbeek/cesm1_data/monthly/"
    elif timing == "daily":
        saveloc = "/home/slingbeek/cesm1_data/daily/"
    else:
        print("Time interval not correct, must be monthly or daily")
        sys.exit()
elif model == "cesm2":
    if timing == "monthly":
        saveloc = "/home/slingbeek/cesm2_data/monthly/"
    elif timing == "daily":
        saveloc = "/home/slingbeek/cesm2_data/daily/"
    else:
        print("Time interval not correct, must be monthly or daily")
        sys.exit()
else:
    print("Model not correct, must be cesm1 or cesm2")
    sys.exit()

PS_input = input("Enter surface pressure file name: ")

@guvectorize(
    "(float64[:], float64[:], float64[:], float32[:])",
    " (n), (n), (m) -> (m)",
    nopython=True,
)
def interp1d_gu(f, x, xi, out):
    """Interpolate field f(x) to xi in ln(x) coordinates."""
    i, imax, x0, f0 = 0, len(xi), x[0], f[0]
    while xi[i]<x0 and i < imax:
        out[i] = np.nan
        i = i + 1
    for x1,f1 in zip(x[1:], f[1:]):
        while xi[i] <= x1 and i < imax:
            out[i] = (f1-f0)/np.log(x1/x0)*np.log(xi[i]/x0)+f0
            i = i + 1
        x0, f0 = x1, f1
    while i < imax:
        out[i] = np.nan
        i = i + 1

print("Loading datasets...")
### Surface pressure
PS_ds = xr.open_dataset(PS_input)
PS = PS_ds.PS

### Open dataset
ds = xr.open_dataset(file)

### New pressure levels
lvls_pt = np.asarray([3.5, 5., 7.5, 10., 15., 23., 33., 43., 53., 63.,
                      74., 88., 103., 122., 143., 168., 200., 233., 274., 322.,
                      380., 440., 500., 590., 681., 763., 821., 850., 887., 913.,
                      936., 957., 976., 993.])

ds = ds.assign_coords({'plev':lvls_pt})
ds['plev'] = ds.plev.assign_attrs(
        {"long_name":"Pressure level","units":"Pa"})

pres = (ds.hyam * ds.P0 + ds.hybm * PS)
ipres = (ds.hyai * ds.P0 + ds.hybi * PS)

### Interpolation for lev or ilev
print("Entering interpolation loop...")
vars3d = [ds[var] for var in ds.data_vars if ds[var].ndim == 4]
for var3d in vars3d:
    if 'lev' in var3d.dims:
        print('Loop for lev entered...')
        for i in range(len(ds[var3d.name]['time'])):
            print("Timestep ", i)
            ds[var3d.name][i] = xr.apply_ufunc(
                interp1d_gu,  var3d[i], pres.sel(time=ds[var3d.name]['time'][i]), ds.plev,
                input_core_dims=[['lev'], ['lev'], ['plev']],
                output_core_dims=[['plev']],
                exclude_dims=set(('lev',)),
                output_dtypes=['float32'],
            ).assign_attrs(var3d[i].attrs)
    else:
        print('Loop for ilev entered...')
        for i in range(len(ds[var3d.name]['time'])):
            print("Timestep ", i)
            ds[var3d.name][i] = xr.apply_ufunc(
                interp1d_gu, var3d[i], ipres.sel(time=ds[var3d.name]['time'][i]), ds.plev,
                input_core_dims=[['ilev'], ['ilev'], ['plev']],
                output_core_dims=[['plev']],
                exclude_dims=set(('ilev',)),
                output_dtypes=['float32'],
            ).assign_attrs(var3d[i].attrs)

del ds['lev']
del ds['hyam']
del ds['hybm']

del ds['ilev']
del ds['hyai']
del ds['hybi']

savename = saveloc + os.path.basename(file)
print(f"Saving {savename}")
ds.to_netcdf(savename)
ds.close()

with open(saveloc+"/logfile.txt", "a") as f:
    f.write("%s: %s \n" %(datetime.now(), os.path.basename(file)))