import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import sys
import glob
import numpy as np
from numba import float32, float64, guvectorize
from datetime import datetime


file, slice = input("Enter file name to split and slice: ").split(",")
# file = input("Enter file name to split: ")

if slice == "start":
    saveloc = "~/paper_data/"
    slices = [slice('2016','2035')]
    names = ["201601-203512"]

if slice == "end":
    saveloc = "~/paper_data/"
    slices = [slice('2080', '2099')]
    names = ["208001-209912"]

else:
    print("Wrong slice, try again.")
    sys.exit()



ds = xr.open_dataset(file)

for i in range(len(slices)):
    stukje = ds.sel(time=slices[i])

    savename = saveloc + os.path.basename(file)[:-16] + names[i] + ".nc"
    print(f"Saving {savename}")
    stukje.to_netcdf(savename)
    stukje.close()
