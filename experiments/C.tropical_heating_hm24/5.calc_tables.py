"""
The easiest way to describe some model differences may be a simple table, 
so this program will calculate any desired values such as z_min/z_max in time.


HEY IDEA: make a plot with y-axis being z anom and x being lead time, then put both pos and neg anoms 
on because they won't interfere due to being on different sides of plot... could make half dotted, but
this is like a table except it's actually readable by the viewer and allows good comparison
"""


from utils import general, vis, model_info
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import xarray as xr
import numpy as np


config_path = Path(__file__).parent / "0.config.yaml"
config = general.read_config(config_path)
exp_dir = Path(config["experiment_dir"]) / config["experiment_name"]
plot_dir = exp_dir / "plots"
n_timesteps = config["n_timesteps"]
summary_datasets = []
for model in config["models"]:
    model_output_path = exp_dir / f"output_{model}.nc"
    ds = xr.open_dataset(model_output_path).squeeze("init_time")
    if ds["lead_time"].values.dtype == "timedelta64[s]":
        ds["lead_time"] = (ds["lead_time"] / np.timedelta64(1, "h")).astype(int)
    heating_path = exp_dir / "auxiliary" / f"heating_{model}.nc"
    heating_ds = xr.open_dataset(heating_path)
    heating = heating_ds["t500"].isel(time=-1).squeeze().values
    lat, lon = ds["lat"].values, ds["lon"].values
    g = 9.81
    model_timestep = model_info.MODEL_TIME_STEP_HOURS[model]
    final_time = n_timesteps * model_timestep
    lead_times = np.arange(0, final_time + 1, model_timestep) # +1 to be inclusive of final lead time
    ds["anom_z500"] = (ds["z500"] - ds["z500"].isel(lead_time=0)) / g # remove mean to get anom, div by g to get height from Z
    ds["anom_z500_min"] = ds["anom_z500"].min(dim=("lat", "lon"))
    ds["anom_z500_max"] = ds["anom_z500"].max(dim=("lat", "lon"))
    summary_ds = ds[["anom_z500_min", "anom_z500_max"]]
    print(f"\nModel: {model}")
    for i, t in enumerate(summary_ds["lead_time"].values):
        print(f"{t}h min/max (m): {summary_ds["anom_z500_min"].sel(lead_time=t):.1f}/{summary_ds["anom_z500_max"].sel(lead_time=t):.1f}")
    
    