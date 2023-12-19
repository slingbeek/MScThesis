#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
import xarray as xr

OVERWRITE = False # overwrite existing files
CALCULATE = True # if False, only list input and output file names

print('overwrite existing files:',OVERWRITE)
print('perform calculations:',CALCULATE)
t1 = time.perf_counter() # start timer

# selecting files to process
filenames = sys.argv[1:] # filenames should be passed as an argument
files = {}
for infile in filenames:
    outfile = infile.replace('.h0.','.h0gm.').replace('.h0zm.','.h0gm.')
    if outfile == infile:
        print(f'skipping: {infile}'); 
        continue
    elif not OVERWRITE and os.path.exists(outfile):
        print(f'skipping: {infile}')
        continue
    else:
        files[infile] = outfile

# calculate global average for every file and save
for i,(infile,outfile) in enumerate(files.items()):
    print(f'{infile} [{i+1}/{len(files)}]\n{outfile}')
    if not CALCULATE: continue
    with xr.open_dataset(infile, decode_times=False) as ds:
        dims = set(ds.dims).intersection({'lat','lon'})
        if 'TREFHT' in ds:
            slat = np.sin(np.deg2rad(ds.lat))
            ds['TREFHT_L1'] = ds.TREFHT * slat
            ds['TREFHT_L2'] = ds.TREFHT * (3 * slat**2 - 1)/2
        dsw = ds.drop('gw').weighted(ds.gw) # weigh by lat weights gw
        dsgm = dsw.mean(dims, keep_attrs=True)
        dsgm.time.attrs = ds.time.attrs # time.attrs somehow gets removed
        dsgm.to_netcdf(outfile)

t2 = time.perf_counter()
print(f"finished in {t2-t1:.3f} seconds",end=" ")
if len(files) >= 1:
    print(f"({(t2-t1)/len(files):.3f} s per file)")
