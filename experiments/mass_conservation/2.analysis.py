import xarray as xr
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
import datetime as dt
import calendar
from matplotlib import colormaps


### Set up and parameter selection ########

# set up paths
this_dir = Path(__file__).parent
model_output_path = this_dir / "data" / "raw_output.nc"
plot_dir = this_dir / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)


# read configuration
config_path = this_dir / "0.config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

ic_dates = [dt.datetime.strptime(str_date, "%Y/%m/%d %H:%M") for str_date in config["ic_dates"]]
n_ics = len(ic_dates)
n_timesteps = config["n_timesteps"]


# set these variables 
config_str = "/glade/work/jmelms/data/dcmip2025_idealized_tests/experiments/long_sim_0/config.yml"
cmap_str = "Set2" # options here: matplotlib.org/stable/tutorials/colors/colormaps.html

# define consts
cp = 1005.0  # J/kg/K
g = 9.81  # m/s^2j
Lv = 2.26e6  # J/kg

def latitude_weighted_mean(da, latitudes):
    """
    Calculate the latitude weighted mean of a variable in a dataset
    """
    lat_radians = np.deg2rad(latitudes)
    weights = np.cos(lat_radians)
    weights.name = "weights"
    var_weighted = da.weighted(weights)
    return var_weighted.mean(dim=["latitude", "longitude"])
##############################################

### Loading the ERA5 data for comparison ###

print(f"Loading data from {model_output_path}")
# path to the ERA5 pressure data
e5_base = "/glade/campaign/collections/rda/data/ds633.0/"
sp_template = f"{e5_base}/e5.oper.an.sfc/{{year:04}}{{month:02}}/e5.oper.an.sfc.128_134_sp.ll025sc.{{year:04}}{{month:02}}0100_{{year:04}}{{month:02}}{{dayend:02}}23.nc"
sp_dict = {}
for time in ic_dates:
    keep_times = time + dt.timedelta(hours=6) * np.arange(n_timesteps+1)
    dayend = calendar.monthrange(time.year, time.month)[1]
    assert keep_times[-1] <= dt.datetime(time.year, time.month, dayend, 23, 0), "Haven't implemented multi-month data yet, try an earlier day or shorter inference"
    sp_file = sp_template.format(year=time.year, month=time.month, dayend=dayend)
    sp_ds = xr.open_dataset(sp_file)
    sp_ds = sp_ds.sel(time=keep_times).squeeze()
    sp_dict[time] = latitude_weighted_mean(sp_ds["SP"], sp_ds["latitude"]) / 100 # convert to hPa
    print(f"Loaded {sp_file} for {time} with shape {sp_ds['SP'].shape}")
    
print(f"Example time:")
print(sp_dict[time])
############################################


### Load model output data ###
# load dataset -- this might be very large, so make sure to request a large enough job
ds = xr.open_dataset(model_output_path).squeeze()  # data only has one member

# convert SP and MSL from Pa to hPa
ds["SP"] = ds["SP"] / 100
ds["MSL"] = ds["MSL"] / 100

# reset time coordinate
time_hours = (ds.time - ds.time[0]) / np.timedelta64(
    1, "h"
)  # set time coord relative to start time
ds.update({"time": time_hours})
ds = ds.assign_attrs({"time units": "hours since start"})

ds["mean_SP"] = latitude_weighted_mean(ds["SP"], ds.latitude)
ds["ens_mean_SP"] = ds["mean_SP"].mean(dim="init_time")
##############################################



### Plot the results ######################
plot_var = "mean_SP"
title = "Global Mean Surface Pressure Trends\nSFNO-Simulated vs. ERA5"
linewidth = 2
fontsize = 24
smallsize = 20
cmap = colormaps.get_cmap(cmap_str)
qual_colors = cmap(np.linspace(0, 1, n_ics))

fig, ax = plt.subplots(figsize=(12.5, 6.5))
sp_mems = ds[plot_var]
for i, ic in enumerate(ic_dates):
    linedat = sp_mems.isel(init_time=i)
    color = qual_colors[i]
    ax.plot(time_hours, linedat, color=color, linewidth=linewidth, label=f"ENS Member {i} (simulated value)")
    ax.plot(time_hours, sp_dict[ic], color=color, linewidth=linewidth, label=f"ENS Member {i} (ERA5 value)", linestyle="--")
    
sp_ens = ds["ens_mean_SP"]
ax.plot(time_hours, sp_ens, color="red", linewidth=2*linewidth, label="Ensemble Mean", linestyle="-")
sp_dict["mean"] = np.array([sp_dict[ic] for ic in ic_dates]).mean(axis=0)
ax.plot(time_hours, sp_dict["mean"], color="red", linewidth=2*linewidth, label="Ensemble Mean (ERA5 value)", linestyle="--")
   
ax.set_xticks(time_hours[::4], (time_hours[::4]/24).values.astype("int"), fontsize=smallsize)
yticks = np.linspace(984, 986, 11)
ax.set_yticks(yticks, yticks, fontsize=smallsize)
ax.set_xlabel("Simulation Time (days)", fontsize=fontsize)
ax.set_ylabel("Pressure (hPa)", fontsize=fontsize)
ax.set_xlim(xmin=0, xmax=time_hours[-1])
fig.suptitle(title, fontsize=30)
ax.grid()
ax.set_facecolor("#ffffff")
fig.tight_layout()
plt.legend(fontsize=12, loc="lower left", ncols=3)
plt.savefig(plot_dir / f"{title}.png", dpi=300, bbox_inches="tight")
###########################################