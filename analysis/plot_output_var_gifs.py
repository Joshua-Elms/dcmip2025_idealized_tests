import xarray as xr
from pathlib import Path
import numpy as np
from matplotlib import colormaps
import yaml
from analysis.plotting_utils import create_and_plot_variable_gif

### set these variables
config_str = "/glade/work/jmelms/data/dcmip2025_idealized_tests/experiments/long_sim_0/config.yml"
cmap_str = "viridis" # options here: matplotlib.org/stable/tutorials/colors/colormaps.html
cmap = colormaps.get_cmap(cmap_str)

### the rest should work itself out
### except for the plot settings at the end
config_path = Path(config_str)
assert config_path.exists(), f"Config file does not exist @ {config_path}"
print(f"Config file exists @ {config_path}")
# import config file
with open(config_path, "r") as stream:
    config = yaml.safe_load(stream)

ic_params = config["initial_condition_parameters"]
keys, val_pairs = zip(*ic_params.items())
vals, units = zip(*val_pairs)
param_val_pairs = dict(zip(keys, vals))
val_types = [isinstance(v, list) for v in vals]
exp_param_idx = val_types.index(
    1
)  # this is the parameter that was varied in the experiment
exp_param = keys[exp_param_idx]
exp_vals = np.array(vals[exp_param_idx])
exp_param_units = units[exp_param_idx]
assert sum(val_types) == 1, "Only one experimental parameter allowed"

exp_dir = (
    Path(config["experiment_dir"]) / config["experiment_name"]
)  # all data for experiment stored here
ic_nc_dir = exp_dir / "ic_nc"  # contains processed ICs in nc format
data_path = exp_dir / "output.nc"  # where output from inference is saved

nt = config["inference_parameters"]["n_steps"]
dt = 6  # hours. all experiments have this timestep
n_expvar = len(exp_vals)
ne = 1  # no ensemble feature has been implemented yet
ensemble_colors = cmap(np.linspace(0, 1, n_expvar))
units_table = {
    "VAR_2T": "K",
    "VAR_10U": "m/s",
    "VAR_10V": "m/s",
    "VAR_100U": "m/s",
    "VAR_100V": "m/s",
    "T": "K",
    "U": "m/s",
    "V": "m/s",
    "Z": "m",
    "R": "%",
    "SP": "hPa",
    "MSL": "hPa",
    "TCW": "kg/m^2",
    "Q": "kg/kg",
}

print(f"Loading data from {data_path}")
print(f"Iter var {exp_param} = {exp_vals} {exp_param_units}")

# load dataset -- this might be very large, so be careful
ds = xr.open_dataset(data_path).isel(ensemble=0)  # data only has one member

# convert SP and MSL from Pa to hPa
ds["SP"] = ds["SP"] / 100
ds["MSL"] = ds["MSL"] / 100

# convert Z geopotential to geopotential height
g = 9.81 # m/s^2
ds["Z"] = ds["Z"] / g

# reset time coordinate
time_hours = (ds.time - ds.time[0]) / np.timedelta64(
    1, "h"
)  # set time coord relative to start time
ds.update({"time": time_hours})
ds = ds.assign_attrs({"time units": "hours since start"})

# set parameters which will remain constant for each iteration of plotting
exp_val_idx = 0
const_params = dict(
    iter_var = "time",
    iter_vals = np.arange(nt),
    adjust = {
        "top": 1,
        "bottom": 0.03,
        "left": 0.13,
        "right": 0.82,
        "hspace": 0.0,
        "wspace": 0.0,
    },
    dpi = 100,
    fps = 4,
    plot_dir = exp_dir / "plots",
    keep_images = True,
    cmap = cmap_str,
)

for exp_val_idx in range(n_expvar):
    title_str = f"{{var_name}} [{{units}}] at {{time}} hours ({exp_param} = {exp_vals[exp_val_idx]} {exp_param_units})"
    for var in list(ds.keys()):
        if "level" in ds[var].dims: # 3D variable
            for i, level in enumerate(ds.level.values):
                select = {exp_param: exp_val_idx, "level": i}
                plot_var = f"{var}_{level}"
                units = units_table[var]
                data = ds[var].isel(select)
                create_and_plot_variable_gif(
                    data,
                    plot_var,
                    units=units,
                    title_str=title_str,
                    **const_params
                )
                print(f"Plotted {plot_var} gif")
        else: # 2D variable
            select = {exp_param: exp_val_idx}
            plot_var = var
            units = units_table[var]
            data = ds[var].isel(select)
        
            create_and_plot_variable_gif(
                    data,
                    plot_var,
                    units=units,
                    title_str=title_str,
                    **const_params
                )
            print(f"Plotted {plot_var} gif")

        