#!/usr/bin/env python3

##############################################################################
# Calculate annual and seasonal mean during selected time periods.
#
# how to run: 
#   >> python timeaverages.py files
# files must contain cf-compliant time series (coord='time') containing
# the periods to be averaged over
##############################################################################
import os
import sys
import time
import numpy as np
import xarray as xr

OVERWRITE = True # overwrite existing files
CALCULATE = True # if False, only list input and output file names
YEARS1 = slice('2020','2039')
YEARS2 = slice('2075','2094')

print('overwrite existing files:',OVERWRITE)
print('perform calculations:',CALCULATE)
t1 = time.perf_counter() # start timer

# selecting files to process
filenames = sys.argv[1:] # filenames should be passed as an argument
files = {}
for infile in filenames:
    outfile = infile.replace('.h0.','.h0tm.').replace('.h0zm.','.h0tm.')
    if outfile == infile:
        print(f'skipping: {infile}'); 
        continue
    elif not OVERWRITE and os.path.exists(outfile):
        print(f'skipping: {infile}')
        continue
    else:
        tstring = outfile.split('.')[-2]
        tstring1 = f'{YEARS1.start}01-{YEARS1.stop}12'
        tstring2 = f'{YEARS2.start}01-{YEARS2.stop}12'
        outfile1 = outfile.replace(tstring,tstring1)
        outfile2 = outfile.replace(tstring,tstring2)
    files[infile] = outfile1, outfile2

# calculate time average for every file and save
wm = lambda ds: ds.weighted(ds.tw).mean('time', keep_attrs=True).drop('tw')
for i,(infile,outfile) in enumerate(files.items()):
    print(f'{infile} [{i+1}/{len(files)}]\n{outfile[0]}\n{outfile[1]}')
    if not CALCULATE: continue
    with xr.open_dataset(infile, decode_times=True) as ds:
        ds = ds.assign_coords(time=('time',ds.time_bnds[:,0].data,{}))
        ds['tw'] = (ds.time_bnds[:,1] - ds.time_bnds[:,0]).dt.days
        ds = ds.drop(('time_bnds','date','datesec','nsteph','ndcur','nscur'))
        dtypes = {v:{'dtype':ds[v].dtype} for v in ds.variables}
        dtypes.pop('tw')
        dtypes.pop('time')
        ds1 = ds.sel(time=YEARS1)
        ds1ann = wm(ds1).assign_coords({'season':'ANN'})
        ds1m = ds1.groupby(ds1.time.dt.season).map(wm)
        ds1c = xr.concat((ds1m, ds1ann), dim='season')
        ds1c.to_netcdf(outfile[0], encoding=dtypes)
        ds2 = ds.sel(time=YEARS2)
        ds2ann = wm(ds2).assign_coords({'season':'ANN'})
        ds2m = ds2.groupby(ds2.time.dt.season).map(wm)
        ds2c = xr.concat((ds2m, ds2ann), dim='season')
        ds2c.to_netcdf(outfile[1], encoding=dtypes)

t2 = time.perf_counter()
print(f"finished in {t2-t1:.3f} seconds",end=" ")
if len(files) >= 1:
    print(f"({(t2-t1)/len(files):.3f} s per file)")
