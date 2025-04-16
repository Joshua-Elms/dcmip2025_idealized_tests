import xarray as xr
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
import datetime as dt
import calendar
from utils import inference, vis
import math
from time import perf_counter
from matplotlib import colormaps


### Set up and parameter selection ########

# set up paths
this_dir = Path(__file__).parent
model_output_path = this_dir / "data" / "output.nc"
olr_save_path = this_dir / "data" / "OLR.nc"
plot_dir = this_dir / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)


# read configuration
config_path = this_dir / "0.config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

ic_dates = [dt.datetime.strptime(str_date, "%Y/%m/%d %H:%M") for str_date in config["ic_dates"]]
n_ics = len(ic_dates)
n_timesteps = config["n_timesteps"]
lead_times_h = np.arange(n_timesteps+1)


# set these variables
cmap_str = "Set2" # options here: matplotlib.org/stable/tutorials/colors/colormaps.html

# define consts
cp = 1005.0  # J/kg/K
g = 9.81  # m/s^2j
Lv = 2.26e6  # J/kg
##############################################


### Load model output & OLR data ###
print(f"Loading model data from {model_output_path}")
ds = xr.open_dataset(model_output_path).squeeze("ensemble", drop=True)

print(f"Loading OLR data from {olr_save_path}")
olr_ds = xr.open_dataset(olr_save_path)
##############################################

### Sanity Test Plots ###

# VAR_2T over range of delta_Ts
title_str = "2m Temperature for $\Delta T={delta_T}$ K"
titles = [title_str.format(delta_T=delta_T) for delta_T in ds["delta_T"].values]

vis.create_and_plot_variable_gif(
    data=ds["VAR_2T"].isel(init_time=0, lead_time=0),
    plot_var="VAR_2T",
    iter_var="delta_T",
    iter_vals=[0, 1, 2],
    plot_dir=plot_dir,
    units="degrees K",
    cmap="magma",
    titles=titles,
    keep_images=True,
    dpi=300,
    fps=1,
)

# VAR_OLR over range of times
titles = [f"OLR at {np.datetime_as_string(time, unit='h')} UTC" for time in olr_ds["init_time"].values]
vis.create_and_plot_variable_gif(
    data=olr_ds["VAR_OLR"],
    plot_var="VAR_OLR",
    iter_var="init_time",
    iter_vals=[0, 1],
    plot_dir=plot_dir,
    units="W/m^2",
    cmap="magma",
    titles=titles,
    keep_images=True,
    dpi=300,
    fps=1,
)
######################################

# ### Plot the results ######################
# plot_var = "MEAN_SP"
# title = "Global Mean Surface Pressure Trends\nSFNO-Simulated vs. ERA5"
# save_title = "sp_trends_sfno_era5.png"
# linewidth = 2
# fontsize = 24
# smallsize = 20
# cmap = colormaps.get_cmap(cmap_str)
# qual_colors = cmap(np.linspace(0, 1, n_ics))

# fig, ax = plt.subplots(figsize=(12.5, 6.5))
# sp_mems = ds[plot_var]
# breakpoint()
# for i, ic in enumerate(ic_dates):
#     linedat = sp_mems.isel(init_time=i)
#     color = qual_colors[i]
#     ax.plot(ds.time, linedat, color=color, linewidth=linewidth, label=f"ENS Member {i} (simulated value)")
#     ax.plot(ds.time, sp_dict[ic], color=color, linewidth=linewidth, label=f"ENS Member {i} (ERA5 value)", linestyle="--")
    
# sp_ens = ds["IC_MEAN_SP"]
# ax.plot(ds.time, sp_ens, color="red", linewidth=2*linewidth, label="Ensemble Mean", linestyle="-")
# sp_dict["mean"] = np.array([sp_dict[ic] for ic in ic_dates]).mean(axis=0)
# ax.plot(ds.time, sp_dict["mean"], color="red", linewidth=2*linewidth, label="Ensemble Mean (ERA5 value)", linestyle="--")
   
# ax.set_xticks(ds.time, (ds.time/24).values.astype("int"), fontsize=smallsize)
# yticks = np.arange(math.floor(sp_mems.min()), math.ceil(sp_mems.max())+0.5, 0.5)
# ax.set_yticks(yticks, yticks, fontsize=smallsize)
# ax.set_xlabel("Simulation Time (days)", fontsize=fontsize)
# ax.set_ylabel("Pressure (hPa)", fontsize=fontsize)
# ax.set_xlim(xmin=0, xmax=ds.time[-1])
# fig.suptitle(title, fontsize=30)
# ax.grid()
# ax.set_facecolor("#ffffff")
# fig.tight_layout()
# plt.legend(fontsize=12, loc="lower left", ncols=3)
# plt.savefig(plot_dir / save_title, dpi=300, bbox_inches="tight")
# ###########################################