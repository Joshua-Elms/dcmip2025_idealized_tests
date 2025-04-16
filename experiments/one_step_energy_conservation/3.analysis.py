import xarray as xr
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
import datetime as dt
from utils import inference, vis
import scipy
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
sb_const = 5.670374419e-8  # W/m^2/K^4, from https://physics.nist.gov/cgi-bin/cuu/Value?sigma
##############################################


### Load model output & OLR data ###
print(f"Loading model data from {model_output_path}")
ds = xr.open_dataset(model_output_path).squeeze("ensemble", drop=True)

print(f"Loading OLR data from {olr_save_path}")
olr_ds = xr.open_dataset(olr_save_path)
##############################################

# ### Sanity Test Plots ###

# # VAR_2T over range of delta_Ts
# title_str = "2m Temperature for $\Delta T={delta_T}$ K"
# titles = [title_str.format(delta_T=delta_T) for delta_T in ds["delta_T"].values]

# vis.create_and_plot_variable_gif(
#     data=ds["VAR_2T"].isel(init_time=0, lead_time=0),
#     plot_var="VAR_2T",
#     iter_var="delta_T",
#     iter_vals=[0, 1, 2],
#     plot_dir=plot_dir,
#     units="degrees K",
#     cmap="magma",
#     titles=titles,
#     keep_images=True,
#     dpi=300,
#     fps=1,
# )

# # VAR_OLR over range of times
# titles = [f"OLR at {np.datetime_as_string(time, unit='h')} UTC" for time in olr_ds["init_time"].values]
# vis.create_and_plot_variable_gif(
#     data=olr_ds["VAR_OLR"],
#     plot_var="VAR_OLR",
#     iter_var="init_time",
#     iter_vals=[0, 1],
#     plot_dir=plot_dir,
#     units="W/m^2",
#     cmap="magma",
#     titles=titles,
#     keep_images=True,
#     dpi=300,
#     fps=1,
# )
# ######################################

### Analysis ###

# Step 1: Calculate the global mean OLR for each init time

olr_ds["MEAN_OLR"] = inference.latitude_weighted_mean(olr_ds["VAR_OLR"], olr_ds["latitude"])

# Step 2: Calculate the effective radiative temperature of the Earth from the OLR
olr_ds["ERT"] = (olr_ds["MEAN_OLR"] / sb_const) ** (1/4)

# Step 3: Calculate the total energy of the atmosphere

### Step 3a: Compute a mask of the p-levels above the surface for integration
sp = ds["SP"]
sp_expanded = sp.expand_dims(dim={"level": ds.sizes["level"]}, axis=-1)
expand_dict = {dim: ds.sizes[dim] for dim in ds.dims if dim != "level"}
levs_expanded = ds["level"].expand_dims(dim=expand_dict)
mask = levs_expanded <= sp_expanded

### Step 3b: Get pressure for integration
pa = 100 * ds.level.values # convert to Pa from hPa, used for integration


### Step 3c: Calculate total energy components
# sensible heat
sensible_heat = cp * ds["T"]
# latent heat - this is already column-integrated
latent_heat = Lv * ds["TCW"]
# geopotential energy
geopotential_energy = g * ds["Z"]
# kinetic energy
kinetic_energy = 0.5 * ds["U"] ** 2 + 0.5 * ds["V"] ** 2

### Step 3d: Calculate total energy by adding all components
# total energy minus latent heat
dry_total_energy = sensible_heat + geopotential_energy + kinetic_energy
dry_total_energy = dry_total_energy.where(mask, np.nan)
# column integration
dry_total_energy_column = (1 / g) * scipy.integrate.trapezoid(
    dry_total_energy, pa, axis=3
)

# sum
ds["VAR_TE"] = (expand_dict, dry_total_energy_column + latent_heat.values)
ds["VAR_TE"] = ds["VAR_TE"].assign_attrs(
    {"units": "J/m^2", "long_name": "Total Energy"}
)

### Step 3e: Weight by latitude
# get latitude weighted total energy (time, ensemble)
ds["LW_TE"] = inference.latitude_weighted_mean(ds["VAR_TE"], ds.latitude)
ds["LW_TE"].assign_attrs(
    {"units": "J/m^2", "long_name": "Latitude-Weighted Total Energy"}
)

### Step 3f: Calculate the time derivative of the total energy
# use second order central difference to calculate the time derivative
# this computes time derivative w/r/t hours, so divide by 3600 to get W/m^2
ds["dLW_TE_dt"] = ds["LW_TE"].differentiate("lead_time") / 3600
ds["dLW_TE_dt"].assign_attrs(
    {"units": "W/m^2", "long_name": "Time Derivative of Latitude-Weighted Total Energy"}
)

breakpoint()


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