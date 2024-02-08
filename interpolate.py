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
elif model == "glens2":
    if timing == "monthly":
        saveloc = "/home/slingbeek/GLENS2/monthly/"
    elif timing == "daily":
        saveloc = "/home/slingbeek/GLENS2/daily/"
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
if model == 'glens2':
    lvls_pt = np.asarray([5.960300e-06, 9.826900e-06, 1.620185e-05, 2.671225e-05, 4.404100e-05, 7.261275e-05,
                          1.197190e-04, 1.973800e-04, 3.254225e-04, 5.365325e-04, 8.846025e-04, 1.458457e-03,
                          2.404575e-03, 3.978250e-03, 6.556826e-03, 1.081383e-02, 1.789800e-02, 2.955775e-02,
                          4.873075e-02, 7.991075e-02, 1.282732e-01, 1.981200e-01, 2.920250e-01, 4.101675e-01,
                          5.534700e-01, 7.304800e-01, 9.559475e-01, 1.244795e+00, 1.612850e+00, 2.079325e+00,
                          2.667425e+00, 3.404875e+00, 4.324575e+00, 5.465400e+00, 6.872850e+00, 8.599725e+00,
                          1.070705e+01, 1.326475e+01, 1.635175e+01, 2.005675e+01, 2.447900e+01, 2.972800e+01,
                          3.592325e+01, 4.319375e+01, 5.167750e+01, 6.152050e+01, 7.375096e+01, 8.782123e+01,
                          1.033171e+02, 1.215472e+02, 1.429940e+02, 1.682251e+02, 1.979081e+02, 2.328286e+02,
                          2.739108e+02, 3.222419e+02, 3.791009e+02, 4.459926e+02, 5.246872e+02, 6.097787e+02,
                          6.913894e+02, 7.634045e+02, 8.208584e+02, 8.595348e+02, 8.870202e+02, 9.126445e+02,
                          9.361984e+02, 9.574855e+02, 9.763254e+02, 9.925561e+02])

else:
    lvls_pt = np.asarray([3.5, 5., 7.5, 10., 15., 23., 33., 43., 53., 63.,
                          74., 88., 103., 122., 143., 168., 200., 233., 274., 322.,
                          380., 440., 500., 590., 681., 763., 821., 850., 887., 913.,
                          936., 957., 976., 993.])

lvls_pt = lvls_pt*100

ds = ds.assign_coords({'plev':lvls_pt})
ds['plev'] = ds.plev.assign_attrs(
        {"long_name":"Pressure level","units":"Pa"})



### Interpolation for lev or ilev
print("Entering interpolation loop...")
vars3d = [ds[var] for var in ds.data_vars if ds[var].ndim == 4]
# for var3d in vars3d:
#     if 'lev' in var3d.dims:
#         print("Calculating pressure for lev...")
#         pres = (ds.hyam * ds.P0 + ds.hybm * PS)
#         for i in range(len(ds[var3d.name]['time'])):
#             print("Timestep ", i)
#             ds[var3d.name][i] = xr.apply_ufunc(
#                 interp1d_gu,  var3d.sel(time=ds[var3d.name]['time'][i]), pres.sel(time=ds[var3d.name]['time'][i]), ds.plev,
#                 input_core_dims=[['lev'], ['lev'], ['plev']],
#                 output_core_dims=[['plev']],
#                 exclude_dims=set(('lev',)),
#                 output_dtypes=['float32'],
#             ).assign_attrs(var3d.sel(time=ds[var3d.name]['time'][i]).attrs)
#     else:
#         print("Calculating pressure for ilev...")
#         ipres = (ds.hyai * ds.P0 + ds.hybi * PS)
#         for i in range(len(ds[var3d.name]['time'])):
#             print("Timestep ", i)
#             ds[var3d.name][i] = xr.apply_ufunc(
#                 interp1d_gu, var3d.sel(time=ds[var3d.name]['time'][i]), ipres.sel(time=ds[var3d.name]['time'][i]), ds.plev,
#                 input_core_dims=[['ilev'], ['ilev'], ['plev']],
#                 output_core_dims=[['plev']],
#                 exclude_dims=set(('ilev',)),
#                 output_dtypes=['float32'],
#             ).assign_attrs(var3d.sel(time=ds[var3d.name]['time'][i]).attrs)

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