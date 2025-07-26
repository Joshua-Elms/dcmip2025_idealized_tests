import xarray as xr
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
import datetime as dt
from utils import inference_sfno
import cdsapi


### Set up and parameter selection ########

# set up paths
this_dir = Path(__file__).parent
save_path = this_dir / "data" / "radiative_imbalance.nc"
data_dir = this_dir / "data" # where to save output from inference
plot_dir = this_dir / "plots"
data_dir.mkdir(parents=True, exist_ok=True) # make dir if it doesn't exist
plot_dir.mkdir(parents=True, exist_ok=True) # make dir if it doesn't exist

dataset = "reanalysis-era5-single-levels"
client = cdsapi.Client()

request = {
    "product_type": ["reanalysis"],
    "variable": ["top_net_thermal_radiation", "top_net_solar_radiation"],
    "year": [2018, 2019],
    "month": [str(month).zfill(2) for month in range(1, 13)],
    "day": [1, 15],
    "time": [0, 12],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

tmp_path = this_dir / "data" / "tmp_isr_olr.nc"
if tmp_path.exists():
    tmp_path.unlink() # remove existing olr file
    
client.retrieve(dataset, request).download(tmp_path) # download the data to disk
ds = xr.open_dataset(tmp_path).squeeze().load() # load the data into memory eagerly
ds = ds.sortby(ds.latitude) # latitude is reversed in the CDS data

# convert to W/m^2, since it's a one-hour accumulation and signed opposite of OLR 
# per https://confluence.ecmwf.int/pages/viewpage.action?pageId=82870405#heading-Meanratesfluxesandaccumulations
ds['VAR_OLR'] = -ds['ttr'] / 3600 # https://codes.ecmwf.int/grib/param-db/179
ds["VAR_ISR"] = ds["tsr"] / 3600 # https://codes.ecmwf.int/grib/param-db/178
ds = ds.reset_coords() # remove some extra coords from ECMWF data
ds = ds[["VAR_ISR", "VAR_OLR"]] # remove all other variables

ds["MEAN_OLR"] = inference_sfno.latitude_weighted_mean(ds["VAR_OLR"], ds["latitude"], device="cuda:0")
ds["MEAN_ISR"] = inference_sfno.latitude_weighted_mean(ds["VAR_ISR"], ds["latitude"], device="cuda:0")
ds["IMBALANCE"] = ds["MEAN_ISR"] - ds["MEAN_OLR"]
ds.to_netcdf(save_path, mode="w", format="NETCDF4", engine="netcdf4") # save to disk

# plot the data
fig, ax1 = plt.subplots(1, 1, figsize=(9, 5))
(-ds["MEAN_OLR"]).plot(ax=ax1, color="blue", linestyle="-", label="OLR")
ds["MEAN_ISR"].plot(ax=ax1, color="red", linestyle="-", label="ISR")
ax1.hlines(0, xmin=ds.valid_time.min(), xmax=ds.valid_time.max(), color="black", linestyle="--", label="y=0")
ax1.set_ylim(-260, 260)
ax1.set_ylabel("Radiation (W/m^2, positive downward)")


ax2 = ax1.twinx()
ds["IMBALANCE"].plot(ax=ax2, label="ISR - OLR", color="green")
ax2.set_ylabel("Imbalance (W/m^2)")
ax1.set_xlabel("Time")
fig.suptitle("Radiative Balance Trend (2018-2019)")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

fig.savefig(plot_dir / "radiative_imbalance.png", dpi=300)


print(f"Saved annual radiative balance data to {save_path}.")