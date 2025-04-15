import xarray as xr
import datetime as dt
import torch
import numpy as np
import logging
from earth2mip import networks # type: ignore
from utils import inference
import dotenv
from pathlib import Path
import numpy as np
import yaml
from time import perf_counter
import warnings

def latitude_weighted_mean(da, latitudes):
    """
    Calculate the latitude weighted mean of a variable in a dataset
    """
    lat_radians = np.deg2rad(latitudes)
    weights = np.cos(lat_radians)
    weights.name = "weights"
    var_weighted = da.weighted(weights)
    return var_weighted.mean(dim=["latitude", "longitude"])

# necessary in case the user requests inference that spans multiple months
    for time in keep_times:
        dayend = calendar.monthrange(time.year, time.month)[1]
        sp_file = sp_template.format(year=time.year, month=time.month, dayend=dayend)
        sp_files.append(sp_file)

for time in keep_times:
        dayend = calendar.monthrange(time.year, time.month)[1]
        sp_file = sp_template.format(year=time.year, month=time.month, dayend=dayend)
        sp_files.append(sp_file)
        
sp_ds = xr.open_mfdataset([sp_files], combine="by_coords", parallel=True)
    sp_ds = sp_ds.sel(time=keep_times).squeeze()