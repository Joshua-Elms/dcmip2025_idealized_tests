import xarray as xr
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
import datetime as dt
from utils import inference, vis
import scipy
from time import perf_counter
import torch
from matplotlib import colormaps


### Set up and parameter selection ########

# set up paths
this_dir = Path(__file__).parent
model_output_path = this_dir / "data" / "output.nc"
rad_save_path = this_dir / "data" / "ISR_OLR.nc" # radiative data
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
device = "cuda:0" if torch.cuda.is_available() else "cpu"


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

print(f"Loading OLR data from {rad_save_path}")
rad_ds = xr.open_dataset(rad_save_path)
##############################################

# ### Sanity Test Plots ###

# VAR_2T over range of delta_Ts
title_str = "2m Temperature for $\Delta T={delta_T}$ K"
titles = [title_str.format(delta_T=delta_T) for delta_T in ds["delta_T"].values]

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

# Step 1: Calculate the global mean ISR/OLR for each init time
rad_ds["MEAN_ISR"] = inference.latitude_weighted_mean(rad_ds["VAR_ISR"], rad_ds["latitude"], device=device)
rad_ds["MEAN_OLR"] = inference.latitude_weighted_mean(rad_ds["VAR_OLR"], rad_ds["latitude"], device=device)

# Step 2: Calculate the effective radiative temperature of the Earth from the OLR
rad_ds["ERT"] = (rad_ds["MEAN_OLR"] / sb_const) ** (1/4)

# Step 3: Calculate the theoretical radiative imbalance of the atmosphere from the effective radiative temperature + delta_T
delta_Ts = np.array(config["temp_deltas_degrees_kelvin"])
# broadcast add delta Ts olr_ds["ERT"] to get starting ERT values
ert = rad_ds["ERT"].expand_dims(dim={"delta_T": ds.sizes["delta_T"]}, axis=-1)
ert = ert + delta_Ts
ert = ert.assign_coords({"delta_T": ds["delta_T"]})
rad_ds["OLR_THEORY"] = sb_const * ert ** 4
rad_ds["IMBALANCE_THEORY"] = rad_ds["MEAN_ISR"] - rad_ds["OLR_THEORY"] # dE/dt = ISR - OLR

# Step 4: Calculate the total energy of the atmosphere

### Step 4a: Compute a mask of the p-levels above the surface for integration
sp = ds["SP"] / 100
sp_expanded = sp.expand_dims(dim={"level": ds.sizes["level"]}, axis=-1)
expand_dict = {dim: ds.sizes[dim] for dim in ds.dims if dim != "level"}
levs_expanded = ds["level"].expand_dims(dim=expand_dict)
mask = levs_expanded <= sp_expanded
ds["terrain_mask"] = mask

### Step 4b: Get pressure for integration
pa = 100 * ds.level.values # convert to Pa from hPa, used for integration

### Step 4c: Calculate total energy components
# sensible heat
sensible_heat = cp * ds["T"]
# latent heat - this is already column-integrated
latent_heat = Lv * ds["TCW"]
# geopotential energy
geopotential_energy = ds["Z"] # geopotential energy is already in J/kg, no need to multiply by g
# kinetic energy
kinetic_energy = 0.5 * ds["U"] ** 2 + 0.5 * ds["V"] ** 2

### Step 4d: Calculate total energy by adding all components
# total energy minus latent heat
dry_total_energy = sensible_heat + geopotential_energy + kinetic_energy
dry_total_energy = dry_total_energy.where(mask, 0.0) # set values below surface to 0
# column integration
print(f"Integrating dry total energy with shape {dry_total_energy.shape} and pa with shape {pa.shape}")
dry_total_energy_column = (1 / g) * scipy.integrate.trapezoid( # add nan-safe version
    dry_total_energy, pa, axis=3
)

# sum
ds["VAR_TE"] = (expand_dict, dry_total_energy_column + latent_heat.values)
ds["VAR_TE"] = ds["VAR_TE"].assign_attrs(
    {"units": "J/m^2", "long_name": "Total Energy"}
)

### Step 4e: Weight by latitude
# get latitude weighted total energy (time, ensemble)
ds["LW_TE"] = inference.latitude_weighted_mean(ds["VAR_TE"], ds.latitude, device=device)
ds["LW_TE"].assign_attrs(
    {"units": "J/m^2", "long_name": "Latitude-Weighted Total Energy"}
)

### Step 4f: Calculate the time derivative of the total energy (OLR proxy)
# energy in J/m^2, so take difference and then div by 21600 to get W/m^2
lw_te = ds["LW_TE"].values
six_hours = 6 * 60 * 60 # seconds
olr_model =  - (lw_te[..., 1] - lw_te[..., 0]) / six_hours # OLR is signed opposite of change in atmospheric energy
# ds["OLR_MODEL"]
# ds["OLR_MODEL"].assign_attrs(
#     {"units": "W/m^2", "long_name": "Time Derivative of Latitude-Weighted Total Energy"}
# )
print(f"Theoretical OLR: {rad_ds['OLR_THEORY'].values}")
print(f"Model OLR: {olr_model}")

fig, ax = plt.subplots(figsize=(10, 8))

# Get the range of values for both axes
min_val = min(rad_ds["OLR_THEORY"].min().item(), np.min(olr_model))
max_val = max(rad_ds["OLR_THEORY"].max().item(), np.max(olr_model))
y_plot_range = [min_val-20, max_val+20]
x_plot_range = [220, 270]


# Plot y=x line
ax.plot(y_plot_range, y_plot_range, 'k-', label='y=x', linewidth=1.5)

# Create scatter plot
scatter = ax.scatter(
    rad_ds["OLR_THEORY"].values.flatten(), 
    olr_model.flatten(),
    c=np.repeat(delta_Ts, len(rad_ds.init_time)),
    cmap='viridis',
    alpha=0.8,
    s=80,
    edgecolor='k',
    linewidth=0.5
)

# Add colorbar
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Temperature Perturbation (ΔT, K)', fontsize=12)

# Labels and title
ax.set_xlabel('Theoretical OLR (W/m²)', fontsize=14)
ax.set_ylabel('Modeled OLR (W/m²)', fontsize=14)
ax.set_title('Comparison of Theoretical vs. Modeled OLR', fontsize=16)

# Equal aspect ratio
# ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.7)

# Set axis limits
ax.set_xlim(x_plot_range)
ax.set_ylim(y_plot_range)

plt.tight_layout()
plt.savefig(plot_dir / "olr_comparison.png", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(5, 3))
# plot lw total energy 
# ax.plot(np.arange(360), ds["LW_TE"].mean(dim="init_time"), label="Model", color="blue")

fig.savefig(plot_dir / "lw_te.png", dpi=300, bbox_inches="tight")

# Plot vertical T profiles
ic = 0
lead_time = 0
T_subset = ds["T"].isel(init_time=ic, lead_time=lead_time)
T_levs = inference.latitude_weighted_mean(T_subset, T_subset.latitude, device="cpu")
fig, ax = plt.subplots(figsize=(5, 3))
for i, delta_T in enumerate(delta_Ts):
    x = T_levs.isel(delta_T=i)
    y = T_levs.level.values  # Reverse the order of levels for plotting p
    ax.plot(x, y, label=f"ΔT={delta_T} K")
    
# ax.set_yscale("log")
ax.set_yscale("log")
ax.invert_yaxis()
ticks = np.concat([y[:6:], y[6::2]])
ax.set_yticks(ticks=ticks, labels=[f"{int(val)}" for val in ticks])
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Pressure (hPa)")
ax.set_title(f"Vertical Temperature Profiles, {ic_dates[ic]}z")
ax.legend(title="Temperature Perturbation (ΔT)", loc="upper right")
fig.savefig(plot_dir / "T_profiles.png", dpi=300, bbox_inches="tight")
breakpoint()

# # Step 5: Test it on ERA5 data

# ic_date = ic_dates[0]
# ds_list = []
# for i in range(10):
#     t = ic_date + dt.timedelta(hours=6*i)
#     test_ds_slice = inference.read_sfno_vars_from_era5_rda(t)
#     ds_list.append(test_ds_slice)
# ds = xr.concat(ds_list, dim="time")

# ### Step 4a: Compute a mask of the p-levels above the surface for integration
# sp = ds["SP"] / 100
# sp_expanded = sp.expand_dims(dim={"level": ds.sizes["level"]}, axis=-1)
# expand_dict = {dim: ds.sizes[dim] for dim in ds.dims if dim != "level"}
# levs_expanded = ds["level"].expand_dims(dim=expand_dict)
# mask = levs_expanded <= sp_expanded

# ### Step 4b: Get pressure for integration
# pa = 100 * ds.level.values # convert to Pa from hPa, used for integration

# ### Step 4c: Calculate total energy components
# # sensible heat
# sensible_heat = cp * ds["T"]
# # latent heat - this is already column-integrated
# latent_heat = Lv * ds["TCW"]
# # geopotential energy
# geopotential_energy = g * ds["Z"]
# # kinetic energy
# kinetic_energy = 0.5 * ds["U"] ** 2 + 0.5 * ds["V"] ** 2

# ### Step 4d: Calculate total energy by adding all components
# # total energy minus latent heat
# dry_total_energy = sensible_heat + geopotential_energy + kinetic_energy
# dry_total_energy = dry_total_energy.where(mask, np.nan)
# # column integration
# dry_total_energy_column = (1 / g) * scipy.integrate.trapezoid(
#     dry_total_energy, pa, axis=1
# )

# # sum
# ds["VAR_TE"] = (expand_dict, dry_total_energy_column + latent_heat.values)
# ds["VAR_TE"] = ds["VAR_TE"].assign_attrs(
#     {"units": "J/m^2", "long_name": "Total Energy"}
# )

# ### Step 4e: Weight by latitude
# # get latitude weighted total energy (time, ensemble)
# ds["LW_TE"] = inference.latitude_weighted_mean(ds["VAR_TE"], ds.latitude)
# ds["LW_TE"].assign_attrs(
#     {"units": "J/m^2", "long_name": "Latitude-Weighted Total Energy"}
# )

# ### Step 4f: Calculate the time derivative of the total energy (OLR proxy)
# # energy in J/m^2, so take difference and then div by 21600 to get W/m^2
# lw_te = ds["LW_TE"].values
# six_hours = 6 * 60 * 60 # seconds
# olr_model =  - (lw_te[..., 1:] - lw_te[..., :-1]) / six_hours # OLR is signed opposite of change in atmospheric energy

# plot terrain mask
# it, dT, lt = 0, 1, 0
# start_lev_idx = 7
# titles = [f"Terrain Mask at {lev}hPa" for lev in ds["level"].values[start_lev_idx:]]
# vis.create_and_plot_variable_gif(
#     data=ds["terrain_mask"].isel(init_time=it, delta_T=dT, lead_time=lt),
#     plot_var="terrain_mask",
#     iter_var="level",
#     iter_vals=np.arange(start_lev_idx, ds.level.size),
#     plot_dir=plot_dir,
#     units="Binary Mask",
#     cmap="binary",
#     titles=titles,
#     keep_images=True,
#     dpi=300,
#     fps=0.2,
# )

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