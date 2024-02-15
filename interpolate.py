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


file, model, timing = input("Enter file name, model and time interval: ").split(",")
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
elif model == "glens2":
    if timing == "monthly":
        saveloc = "/home/slingbeek/GLENS2/monthly/"
    elif timing == "daily":
        saveloc = "/home/slingbeek/GLENS2/daily/"
    else:
        print("Time interval not correct, must be monthly or daily")
        sys.exit()
# elif model == "cmip6":
#     if timing == "monthly":
#         saveloc = "/home/slingbeek/CMIP6/monthly/"
#     elif timing == "daily":
#         saveloc = "/home/slingbeek/CMIP6/daily/"
#     else:
#         print("Time interval not correct, must be monthly or daily")
#         sys.exit()
else:
    print("Model not correct, must be cesm1, cesm2 or glens2")
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
if model == "cmip6":
    PS = PS_ds.ps
else:
    PS = PS_ds.PS

### Open dataset
ds = xr.open_dataset(file)

### New pressure levels
if model == 'glens2':
    lvls_pt = np.asarray([6.0e-06, 9.8e-06, 1.6e-05, 2.7e-05, 4.4e-05, 7.3e-05, 1.2e-04, 2.0e-04, 3.3e-04, 5.4e-04,
                          8.8e-04, 1.5e-03, 2.4e-03, 4.0e-03, 6.6e-03, 1.1e-02, 1.8e-02, 3.0e-02, 4.9e-02, 8.0e-02,
                          1.3e-01, 2.0e-01, 2.9e-01, 4.1e-01, 5.5e-01, 7.3e-01, 9.6e-01, 1.2e+00, 1.6e+00, 2.1e+00,
                          2.7e+00, 3.4e+00, 4.3e+00, 5.5e+00, 6.9e+00, 8.6e+00, 1.1e+01, 1.3e+01, 1.6e+01, 2.0e+01,
                          2.4e+01, 3.0e+01, 3.6e+01, 4.3e+01, 5.2e+01, 6.2e+01, 7.4e+01, 8.8e+01, 1.0e+02, 1.2e+02,
                          1.4e+02, 1.7e+02, 2.0e+02, 2.3e+02, 2.7e+02, 3.2e+02, 3.8e+02, 4.5e+02, 5.2e+02, 6.1e+02,
                          6.9e+02, 7.6e+02, 8.2e+02, 8.6e+02, 8.9e+02, 9.1e+02, 9.4e+02, 9.6e+02, 9.8e+02, 9.9e+02])

else:
    lvls_pt = np.asarray([3.5, 5., 7.5, 10., 15., 23., 33., 43., 53., 63., 74., 88., 103., 122., 143., 168., 200.,
                          233., 274., 322., 380., 440., 500., 590., 681., 763., 821., 850., 887., 913., 936., 957.,
                          976., 993.])
    lvls_pt = lvls_pt*100

ds = ds.assign_coords({'plev':lvls_pt})
ds['plev'] = ds.plev.assign_attrs(
        {"long_name":"Pressure level","units":"Pa"})



### Interpolation for lev or ilev
print("Entering interpolation loop...")
vars3d = [ds[var] for var in ds.data_vars if ds[var].ndim == 4]

for var3d in vars3d:
    if 'lev' in var3d.dims:
        print("Calculating pressure for lev...")
        pres = (ds.P0 * ds.hyam + PS * ds.hybm)
        pres.compute()
        print("Applying ufunc...")
        ds[var3d.name] = xr.apply_ufunc(
            interp1d_gu,  var3d, pres, ds.plev,
            input_core_dims=[['lev'], ['lev'], ['plev']],
            output_core_dims=[['plev']],
            exclude_dims=set(('lev',)),
            output_dtypes=['float32'],
        ).assign_attrs(var3d.attrs)
    else:
        print("Calculating pressure for ilev...")
        ipres = (ds.P0 * ds.hyai + PS * ds.hybi)
        ipres.compute()
        print("Applying ufunc...")
        ds[var3d.name] = xr.apply_ufunc(
            interp1d_gu, var3d, ipres.sel, ds.plev,
            input_core_dims=[['ilev'], ['ilev'], ['plev']],
            output_core_dims=[['plev']],
            exclude_dims=set(('ilev',)),
            output_dtypes=['float32'],
        ).assign_attrs(var3d.attrs)

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