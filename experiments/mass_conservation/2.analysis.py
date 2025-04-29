import xarray as xr
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
import datetime as dt
import calendar
from utils import inference
import math
from time import perf_counter
from matplotlib import colormaps


### Set up and parameter selection ########

# set up paths
this_dir = Path(__file__).parent
model_output_path = this_dir / "data" / "output.nc"
era5_surface_pressure_path = this_dir / "data" / "era5_surface_pressure_cache.nc"
plot_dir = this_dir / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)
e5_base = "/glade/campaign/collections/rda/data/ds633.0/"



# read configuration
config_path = this_dir / "0.config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

ic_dates = [dt.datetime.strptime(str_date, "%Y/%m/%d %H:%M") for str_date in config["ic_dates"]]
n_ics = len(ic_dates)
n_timesteps = config["n_timesteps"]


# set these variables
cmap_str = "Set2" # options here: matplotlib.org/stable/tutorials/colors/colormaps.html

# define consts
cp = 1005.0  # J/kg/K
g = 9.81  # m/s^2j
Lv = 2.26e6  # J/kg
##############################################

### Loading the ERA5 data for comparison ###

# check to see if the data is already cached
cached = True
if era5_surface_pressure_path.exists():
    e5_sp_ds = xr.open_dataset(era5_surface_pressure_path)
    # check whether the cached data has at least as many timesteps as the model output
    if len(e5_sp_ds.init_time) < n_ics:
        print(f"Cached data has only {len(e5_sp_ds.init_time)} timesteps, will re-download.")
        cached = False
        
    # check whether the cached data has the same initial conditions as the model output
    for ic_date in ic_dates:
        # check whether all dates are in the cached data
        if ic_date not in e5_sp_ds.init_time.values:
            print(f"Missing data for {ic_date}, will re-download.")
            cached = False
            break
    
    print(f"Loaded cached data from {era5_surface_pressure_path} with {len(e5_sp_ds.init_time)} timesteps.")
    
if not cached:
    print(f"Loading ERA5 pressure data from {e5_base}")

    # path to the ERA5 pressure data
    sp_template = f"{e5_base}/e5.oper.an.sfc/{{year:04}}{{month:02}}/e5.oper.an.sfc.128_134_sp.ll025sc.{{year:04}}{{month:02}}0100_{{year:04}}{{month:02}}{{dayend:02}}23.nc"
    sp_dict = {}
    for ic_date in ic_dates:
        keep_times = ic_date + dt.timedelta(hours=6) * np.arange(n_timesteps+1)
        sp_files = []
        # necessary in case the user requests inference that spans multiple months
        year_months = set((t.year, t.month) for t in keep_times)
        for year, month in year_months:
            dayend = calendar.monthrange(year, month)[1]
            sp_file = sp_template.format(year=year, month=month, dayend=dayend)
            sp_files.append(sp_file)

        sp_ds = xr.open_mfdataset(sp_files, parallel=True)
        sp_ds = sp_ds.sel(time=keep_times).squeeze()
        sp_dict[ic_date] = inference.latitude_weighted_mean(sp_ds["SP"], sp_ds["latitude"]) / 100 # convert to hPa
        print(f"Loaded {sp_file} for {ic_date} with shape {sp_ds['SP'].shape}")
    
    print(f"Caching the ERA5 data for {len(ic_dates)} initial conditions.")
    # combine the data into a single xarray dataset
    e5_sp_ds = xr.concat([sp_dict[ic] for ic in ic_dates], dim="init_time")
    e5_sp_ds["mean"] = e5_sp_ds.mean(dim="init_time")

    # cache the data for quicker visualizations in the future
    e5_sp_ds.to_netcdf(era5_surface_pressure_path, mode="w", format="NETCDF4", engine="netcdf4")
############################################


### Load model output data ###
print(f"Loading data from {model_output_path}")
# load dataset -- this might be very large, so make sure to request a large enough job
ds = xr.open_dataset(model_output_path)

# reset time coordinate
# time_hours = ds.time * np.timedelta64(
#     1, "h"
# )  # set time coord relative to start time
# ds.update({"time": time_hours})
time_hours = np.arange(0, n_timesteps + 1) * 6
ds = ds.assign_attrs({"time units": "hours since start"})
##############################################


### benchmarking ###
n_iters = 2
dummy_storage = []
start = perf_counter()
for _ in range(n_iters):
    dummy_storage.append(inference.slow_latitude_weighted_mean(ds["SP"], ds.latitude))
end = perf_counter()
elapsed_0 = (end - start) / n_iters
print(f"Elapsed time for {n_iters} iterations of latitude_weighted_mean: {elapsed_0:.4f} seconds")

start = perf_counter()
for _ in range(n_iters):
    dummy_storage.append(inference.latitude_weighted_mean(ds["SP"], ds.latitude))
end = perf_counter()
elapsed_1 = (end - start) / n_iters
print(f"Elapsed time for {n_iters} iterations of alt_latitude_weighted_mean: {elapsed_1:.4f} seconds")
print(f"Speedup: {elapsed_0/elapsed_1:.2f}x")
####################

### Plot the results ######################
plot_var = "MEAN_SP"
title = "Global Mean Surface Pressure Trends\nSFNO-Simulated vs. ERA5"
save_title = "sp_trends_sfno_era5.png"
linewidth = 2
fontsize = 24
smallsize = 20
cmap = colormaps.get_cmap(cmap_str)
qual_colors = cmap(np.linspace(0, 1, n_ics))

fig, ax = plt.subplots(figsize=(12.5, 6.5))
sp_mems = ds[plot_var]
some_var = None
for i, ic in enumerate(ic_dates):
    breakpoint()
    linedat = sp_mems.isel(init_time=i)
    color = qual_colors[i]
    ax.plot(time_hours, linedat, color=color, linewidth=linewidth, label=f"ENS Member {i} (simulated value)")
    ax.plot(time_hours, some_var, color=color, linewidth=linewidth, label=f"ENS Member {i} (ERA5 value)", linestyle="--")
    
sp_ens = ds["IC_MEAN_SP"]
ax.plot(time_hours, sp_ens, color="red", linewidth=2*linewidth, label="Ensemble Mean", linestyle="-")
ax.plot(time_hours, e5_sp_ds["mean"], color="red", linewidth=2*linewidth, label="Ensemble Mean (ERA5 value)", linestyle="--")
   
ax.set_xticks(time_hours[::14*7], (time_hours[::14*7]/(6*28)), fontsize=smallsize)
yticks = np.arange(math.floor(sp_mems.min()), math.ceil(sp_mems.max())+0.5, 0.5)
ax.set_yticks(yticks, yticks, fontsize=smallsize)
ax.set_xlabel("Simulation Time (weeks)", fontsize=fontsize)
ax.set_ylabel("Pressure (hPa)", fontsize=fontsize)
ax.set_xlim(xmin=0, xmax=time_hours[-1])
fig.suptitle(title, fontsize=30)
ax.grid()
ax.set_facecolor("#ffffff")
fig.tight_layout()
plt.legend(fontsize=12, loc="lower left", ncols=3)
plt.savefig(plot_dir / save_title, dpi=300, bbox_inches="tight")
###########################################