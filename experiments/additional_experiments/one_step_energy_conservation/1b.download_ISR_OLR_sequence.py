"""
Important: I'm about to kludge this code together to get it 
to work just in time for my CASCADE presentation. This won't 
be pretty, but it will work. I will clean it up later.

Needs to be updated to permit downloading a non-certain 
sequence of lead times.
"""

import xarray as xr
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
import datetime as dt
from utils import inference
import cdsapi


### Set up and parameter selection ########

# set up paths
this_dir = Path(__file__).parent
save_path = this_dir / "data" / "ISR_OLR_sequence.nc"
data_dir = this_dir / "data" # where to save output from inference
data_dir.mkdir(parents=True, exist_ok=True) # make dir if it doesn't exist

# read configuration
config_path = this_dir / "0.config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    
ic_dates = [dt.datetime.strptime(str_date, "%Y/%m/%d %H:%M") for str_date in config["ic_dates"]]
n_ics = len(ic_dates)
assert config["n_timesteps"] == 27, "Must be one-week forecasts for this experiment"
lead_times_h = np.arange(0, 6*config["n_timesteps"]+1, 6)

dataset = "reanalysis-era5-single-levels"
client = cdsapi.Client()
n_downloads = len(ic_dates)
print(f"Downloading ISR & OLR data for {n_downloads} ICs.")
ds_outer = []
for i, ic_date in enumerate(ic_dates):
    t = ic_date
    assert (t.day == 1) & (t.hour == 0), f"Only the first day of the month and 00:00 UTC are supported, not {t}."
    print(f"Downloading ISR & OLR data for IC {ic_date} and lead time {28*6} hours ({i+1}/{n_downloads}).")
    request = {
    "product_type": ["reanalysis"],
    "variable": ["top_net_thermal_radiation", "top_net_solar_radiation"],
    "year": [str(t.year)],
    "month": [str(t.month).zfill(2)],
    "day": [str(day).zfill(2) for day in np.arange(1, 8)],
    "time": [0, 6, 12, 18],
    "data_format": "netcdf",
    "download_format": "unarchived"
    }

    tmp_path = this_dir / "data" / "tmp_isr_olr_sequence.nc"
    if tmp_path.exists():
        tmp_path.unlink() # remove existing olr file
    client.retrieve(dataset, request).download(tmp_path) # download the data to disk
    ds_inner = xr.open_dataset(tmp_path).squeeze().load() # load the data into memory eagerly
    ds_inner = ds_inner.sortby(ds_inner.latitude) # latitude is reversed in the CDS data
    ds_inner = ds_inner.assign_coords({"valid_time": lead_times_h})
    ds_inner = ds_inner.rename({"valid_time": "lead_time"})
    tmp_path.unlink()
    
    ds_outer.append(ds_inner)
ds = xr.concat(ds_outer, dim="init_time")
ds = ds.assign_coords({"init_time": ic_dates})

# convert to W/m^2, since it's a one-hour accumulation and signed opposite of OLR 
# per https://confluence.ecmwf.int/pages/viewpage.action?pageId=82870405#heading-Meanratesfluxesandaccumulations
ds['VAR_OLR'] = -ds['ttr'] / 3600 # https://codes.ecmwf.int/grib/param-db/179
ds["VAR_ISR"] = ds["tsr"] / 3600 # https://codes.ecmwf.int/grib/param-db/178
ds = ds.reset_coords() # remove some extra coords from ECMWF data
ds = ds[["VAR_ISR", "VAR_OLR"]] # remove all other variables

ds.to_netcdf(save_path, mode="w", format="NETCDF4", engine="netcdf4") # save to disk
print(f"Saved ISR & OLR (long sequence) data to {save_path}.")