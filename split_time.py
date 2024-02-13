import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import sys
import glob
import numpy as np
from numba import float32, float64, guvectorize
from datetime import datetime


file, model = input("Enter file name to split and model: ").split(",")
if model == "glens2":
    saveloc = "~/GLENS2/monthly"
    slices = [slice('2080', '2089'), slice('2090', '2100')]
    names = ["208001-208912", "209001-210012"]

elif model == "cmip6":
    saveloc = "~/CMIP6/monthly"
    slices = [slice('2015', '2019'), slice('2020', '2029'), slice('2030', '2039'), slice('2080', '2089'),
              slice('2090', '2100')]
    names = ["201501-201912", "202001-202912", "203001-203912", "208001-208912", "209001-210012"]
else:
    print("Wrong model, try again.")
    sys.exit()



ds = xr.open_dataset(file)

for i in range(len(slices)):
    stukje = ds.sel(time=slices[i])

    savename = saveloc + os.path.basename(file)[:-16] + names[i] + ".nc"
    print(f"Saving {savename}")
    stukje.to_netcdf(savename)
    stukje.close()
