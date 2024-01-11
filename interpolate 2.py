#!/usr/bin/python
"""This module interpolates the data from hybrid to isentropic levels.

This file will read the Harmonie data on hybrid levels and interpolate
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

p_ref = 100000
cp = 1004
Rdry = 286.9
kappa = Rdry/cp
dtdp_crit = -2e-4 # ignoring layers where dtheta/dp > dtdp_crit
lvls_pt = (np.cumsum([0] + [10/6] * 39 + [10/4] * 6 + [10/2] * 3 + [10] 
    + [50] * 2 + [100] * 3) + 295).astype('float64') # new theta levels
files = sorted(glob.glob("../Data/LambertGrid/629x989/fc2017090512+???.nc"))
saveloc = "../Data/LambertGrid/629x989interped/"

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

@guvectorize(
    "(float64[:], float64[:], float64[:], float32[:])",
    " (n), (n), (m) -> (m)",
    nopython=True,
)
def interp1d_pres_gu(p, x, xi, out):
    """Interpolate field p(x) to xi in ln(x) coordinates."""
    i, imax, x0, p0 = 0, len(xi), x[0], p[0]
    while xi[i]<x0 and i < imax:
        out[i] = np.nan      
        i = i + 1 
    for x1,p1 in zip(x[1:], p[1:]):
        while xi[i] <= x1 and i < imax:
            gamma = np.log(p1/p0)/np.log(x1/x0)
            pi = p1 * (xi[i]/x1)**gamma
            out[i] = pi
            i = i + 1
        x0, p0 = x1, p1
    while i < imax:
        out[i] = np.nan
        i = i + 1
        
@guvectorize(
    "(float64[:], float64[:], float64[:])",
    " (n), (n) -> (n)",
    nopython=True,
)
def calc_dtdp_gu(t, p, dtdp):
    """Calculate dthetadp on hybrid levels."""
    i, imax, ln = 1, len(t), np.log
    dtdp[0] = (ln(t[2]/t[0])/ln(p[2]/p[0]) - ln(t[2]/t[1])/ln(p[2]/p[1]) 
               + ln(t[1]/t[0])/ln(p[1]/p[0])) * t[0]/p[0]
    if dtdp[0] > dtdp_crit: # failure of parabola fitting
        dtdp[0] = (t[1]-t[0])/(p[1]-p[0])
    while not np.isnan(t[i+1]) and i < (imax - 1):
        dtdp[i] =  t[i]/p[i] * (
            ln(p[i]/p[i-1])*ln(t[i+1]/t[i])/(ln(p[i+1]/p[i])*ln(p[i+1]/p[i-1]))
            + ln(p[i+1]/p[i])*ln(t[i]/t[i-1])/(ln(p[i]/p[i-1])*ln(p[i+1]/p[i-1]))
        )
        i = i + 1
    dtdp[i] = (ln(t[i]/t[i-2])/ln(p[i]/p[i-2]) + ln(t[i]/t[i-1])/ln(p[i]/p[i-1]) 
               - ln(t[i-1]/t[i-2])/ln(p[i-1]/p[i-2])) * t[i]/p[i]
    i = i + 1
    while i < imax:
        dtdp[i] = np.nan
        i = i + 1

def argsort3d(da, dim):
    m, n, k = da.shape
    ids = np.ogrid[:m,:n,:k]
    ax = da.dims.index(dim)
    ids[ax] = da.argsort(ax)
    return tuple(ids)

def stabilize(ds, theta, pres):
    """Mask theta values where dtheta/dp > dtdp_crit"""
    assert theta.dims == ('hybrid','y','x')
    assert pres.dims == ('hybrid','y','x')
    vars3d = [var for var in ds.data_vars if 'hybrid' in ds[var].dims and ds[var].ndim == 3]
    y,x = np.mgrid[:theta.shape[1],:theta.shape[2]]
    i0 = np.zeros(theta.shape[1:], dtype=int) 
    for i1 in range(1,len(theta)):
        t0, p0 = theta.data[i0,y,x], pres.data[i0,y,x]
        t1, p1 = theta.data[i1,:,:], pres.data[i1,:,:]
        dtdp = ((t1-t0)/(p1-p0))
        isInvalid = dtdp > dtdp_crit
        theta.data[i1][isInvalid] = np.nan
        for var3d in vars3d:
            ds[var3d].data[i1][isInvalid] = np.nan
        i0[~isInvalid] = i1
    return ds, theta

for file in files:
    ds = xr.open_dataset(file)
    pres = (ds.a + ds.b * ds.p0m)
    theta = (ds.t * (p_ref/pres)**kappa)
    ds, theta = stabilize(ds, theta, pres)
    ids = argsort3d(theta, 'hybrid')
    theta[:] = theta.data[ids]
    pres[:] = pres.data[ids]
    dtdp = xr.apply_ufunc(
        calc_dtdp_gu, theta, pres,
        input_core_dims=[['hybrid'], ['hybrid']], 
        output_core_dims=[['hybrid']], 
        output_dtypes=[theta.dtype],
    ).assign_attrs({'units':'K/Pa','long_name':'dtheta/dp'})
    ds['dtdp'] = dtdp#.transpose('hybrid','y','x')
    ds = ds.assign_coords({'theta':lvls_pt})
    ds['theta'] = ds.theta.assign_attrs(
        {"long_name":"potential temperature","units":"K"})
    vars3d = [ds[var] for var in ds.data_vars if ds[var].ndim==3]
    for var3d in vars3d:
        var3d = var3d.astype('float64').transpose('hybrid','y','x')
        if var3d.name != 'dtdp':
            var3d[:] = var3d.data[ids]
        ds[var3d.name] = xr.apply_ufunc(
            interp1d_gu,  var3d, theta, ds.theta,
            input_core_dims=[['hybrid'], ['hybrid'], ['theta']], 
            output_core_dims=[['theta']], 
            exclude_dims=set(('hybrid',)),  
            output_dtypes=['float32'],
        ).transpose('theta','y','x').assign_attrs(var3d.attrs)
    ds['pres'] = xr.apply_ufunc(
        interp1d_pres_gu,  pres, theta, ds.theta,
        input_core_dims=[['hybrid'], ['hybrid'], ['theta']], 
        output_core_dims=[['theta']], 
        exclude_dims=set(('hybrid',)),  
        output_dtypes=['float32'],
    ).transpose('theta','y','x').assign_attrs({'units':'Pa','long_name':'Pressure'})
    del ds['a']
    del ds['b']
    del ds['hybrid']
    savename = saveloc + os.path.basename(file)
    print(f"Saving {savename}")
    ds.to_netcdf(savename)
    ds.close()
