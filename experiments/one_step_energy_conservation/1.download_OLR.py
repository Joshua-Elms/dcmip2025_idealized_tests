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
olr_save_path = this_dir / "data" / "OLR.nc"


# read configuration
config_path = this_dir / "0.config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    
ic_dates = [dt.datetime.strptime(str_date, "%Y/%m/%d %H:%M") for str_date in config["ic_dates"]]
n_ics = len(ic_dates)

dataset = "reanalysis-era5-single-levels"
client = cdsapi.Client()

ds_list = [] # collect datasets to concatenate later

for i, ic_date in enumerate(ic_dates):
    
    request = {
        "product_type": ["reanalysis"],
        "variable": ["top_net_thermal_radiation"],
        "year": [str(ic_date.year)],
        "month": [str(ic_date.month).zfill(2)],
        "day": [str(ic_date.day).zfill(2)],
        "time": [f"{str(ic_date.hour).zfill(2)}:00"],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }

    tmp_olr_path = this_dir / "data" / "tmp_olr.nc"
    if tmp_olr_path.exists():
        tmp_olr_path.unlink() # remove existing olr file
        
    client.retrieve(dataset, request).download(tmp_olr_path) # download the data to disk
    ds = xr.open_dataset(tmp_olr_path).squeeze().load() # load the data into memory eagerly
    ds = ds.sortby(ds.latitude) # latitude is reversed in the CDS data
    ds_list.append(ds)    
    
    tmp_olr_path.unlink() # remove the olr file after download

# concatenate the datasets along the time dimension
ds = xr.concat(ds_list, dim="init_time")
ds = ds.assign_coords({"init_time": ic_dates})
# convert to W/m^2, since it's a one-hour accumulation and signed opposite of OLR 
# per https://confluence.ecmwf.int/pages/viewpage.action?pageId=82870405#heading-Meanratesfluxesandaccumulations
ds['VAR_OLR'] = -ds['ttr'] / 3600 
ds = ds.drop_vars(['ttr']) # remove the original variable

ds.to_netcdf(olr_save_path, mode="w", format="NETCDF4", engine="netcdf4") # save to disk
print(f"Saved OLR data to {olr_save_path}.")