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


print_data = False
tick_interval_timesteps = 8 # 2 days
config_path = Path(__file__).parent / "0.config.yaml"
config = general.read_config(config_path)
exp_dir = Path(config["experiment_dir"]) / config["experiment_name"]
plot_dir = exp_dir / "plots"
n_timesteps = config["n_timesteps"]
summary_datasets = []
for model in config["models"]:
    print(f"\nModel: {model}")
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
    
    if print_data:
        print(f"Displaying anomalies")
        for i, t in enumerate(summary_ds["lead_time"].values):
            print(f"{t}h min/max (m): {summary_ds["anom_z500_min"].sel(lead_time=t):.1f}/{summary_ds["anom_z500_max"].sel(lead_time=t):.1f}")
            
    summary_datasets.append(summary_ds)
    
### Figure 1: z500 anom mag over lead times ###

full_ds = xr.concat(summary_datasets, dim="model")
print(full_ds)

fig, ax = plt.subplots(figsize=(8,5))
colors = plt.get_cmap("tab20").colors
ax_mag = max(abs(full_ds["anom_z500_max"].values.max()), abs(full_ds["anom_z500_max"].values.min()))
ax_mag = 700
ax.set_ylim(bottom=-ax_mag, top=ax_mag)
ax.set_xlabel("Lead Time (days)")
all_xticks = [hours / 24 for hours in range(0, (n_timesteps * 6) + 1, 6)]
major_xticks = [time for i, time in enumerate(all_xticks) if i % tick_interval_timesteps == 0]
minor_xticks = [time for i, time in enumerate(all_xticks) if i % tick_interval_timesteps != 0]
major_xtick_labels = [(int(time) if (int(time) == time) else time) for time in major_xticks]
minor_xtick_labels = ["" for tick in minor_xticks]
ax.set_xticks(ticks=major_xticks, labels=major_xtick_labels, minor=False)
ax.set_xticks(ticks=minor_xticks, labels=minor_xtick_labels, minor=True)
ax.set_ylabel("z500 anomaly (m)")
ax.hlines(y=0, xmin=0, xmax=(n_timesteps/4), linestyle="dashed", colors="black")

for i, model in enumerate(config["models"]):
    model_timestep = model_info.MODEL_TIME_STEP_HOURS[model]
    final_time = n_timesteps * model_timestep
    lead_times = np.arange(0, final_time + 1, model_timestep) # +1 to be inclusive of final lead time
    model_maxs = full_ds["anom_z500_max"].sel(model=model, lead_time=lead_times).squeeze().values
    model_mins = full_ds["anom_z500_min"].sel(model=model, lead_time=lead_times).squeeze().values
    assert len(model_maxs.shape) == 1, f"Data has wrong number of dimensions: {len(model_maxs.shape)}"
    
    max_linecolor = colors[i*2]
    min_linecolor = colors[(i*2) + 1]
    
    # if 
    ax.plot(lead_times/24, model_maxs, color=max_linecolor, label=f"{model} max")
    ax.plot(lead_times/24, model_mins, color=min_linecolor, label=f"{model} min")
    
ax.set_xlim(left=0, right=(n_timesteps*6)/24)
ax.legend(loc="upper left", ncols=4, fontsize=6)
ax.grid(visible=True, which="major", color="grey", alpha=0.3)
fig.savefig(plot_dir/"z500_anom_mag_compare.png", dpi=300)
print("Saved anom mag comparison")


    
    