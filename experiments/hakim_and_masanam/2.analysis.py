import xarray as xr
import datetime as dt
import numpy as np
from utils import vis
from pathlib import Path
import numpy as np
import yaml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# set up paths
this_dir = Path(__file__).parent
data_dir = this_dir / "data" # where to save output from inference
data_path = data_dir / "output.nc"
plot_dir = this_dir / "plots" # save figures here
plot_dir.mkdir(parents=True, exist_ok=True) # make dir if it doesn't exist

# read configuration
config_path = this_dir / "0.config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    
# convenience vars
n_timesteps = config["n_timesteps"]
lead_times_h = np.arange(0, 6*n_timesteps+1, 6)
g = 9.81 # m/s^2

# load dataset
ds = xr.open_dataset(data_path)

# make pretty plots
titles = [f"VAR_2T at t={t*6} hours" for t in range(n_timesteps+1)]
vis.create_and_plot_variable_gif(
    data=ds["VAR_2T"].isel(ensemble=0),
    plot_var="VAR_2T",
    iter_var="lead_time",
    iter_vals=np.arange(0, n_timesteps+1),
    plot_dir=plot_dir,
    units="degrees K",
    cmap="magma",
    titles=titles,
    keep_images=False,
    dpi=300,
    fps=8,
)

print(f"GIF saved to {plot_dir}/VAR_2T.gif.")

titles = [f"Z500 at t={t*6} hours" for t in range(n_timesteps+1)]
vis.create_and_plot_variable_gif(
    data=ds["Z"].isel(ensemble=0).sel(level=500)/(10*g),
    plot_var="Z",
    iter_var="lead_time",
    iter_vals=np.arange(0, n_timesteps+1),
    plot_dir=plot_dir,
    units="dam",
    cmap="coolwarm",
    titles=titles,
    keep_images=False,
    dpi=300,
    fps=8,
)

print(f"GIF saved to {plot_dir}/Z.gif.")

# load the tendency dataset
tds_path = this_dir / config["IC_tendency_path"]
tds = xr.open_dataset(tds_path)

# debug tds
titles = ["DJF VAR_2T Tendency"]
vis.create_and_plot_variable_gif(
    data=tds["VAR_2T"],
    plot_var="VAR_2T_tendency",
    iter_var="ensemble",
    iter_vals=[0],
    plot_dir=plot_dir,
    units="degrees K",
    cmap="magma",
    titles=titles,
    keep_images=True,
    dpi=300,
    fps=1,
)
