import xarray as xr
import datetime as dt
import torch
import numpy as np
import logging
from earth2mip import networks # type: ignore
from utils import inference
import dotenv
from pathlib import Path
import numpy as np
import yaml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# set up paths
this_dir = Path(__file__).parent
data_dir = this_dir / "data" # where to save output from inference
output_path = data_dir / "output.nc"

# read configuration
config_path = this_dir / "0.config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    
# load the earth2mip environment variables
dotenv.load_dotenv()

# load the model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}.")
model = networks.get_model("fcnv2_sm").to(device)
print("Model loaded.")

# set data paths
IC_path = Path(config["IC_path"])
IC_tendency_path = Path(config["IC_tendency_path"])

# convenience vars
n_timesteps = config["n_timesteps"]
lead_times_h = np.arange(0, 6*n_timesteps+1, 6)

# read unperturbed initial conditions
# TODO: fix the latitude in upstream code
IC_ds = xr.open_dataset(IC_path).sortby("latitude", ascending=False)

# if file exists tendency_path at tendency_path, load it
if IC_tendency_path.exists():
    print(f"Loading cached tendencies from {IC_tendency_path}.")
    tds = xr.open_dataset(IC_tendency_path).sortby("latitude", ascending=False)
# else, compute tendencies and save to tendency_path for future use
else:
    print(f"Computing tendencies and saving to {IC_tendency_path}.")
    tendency_ds = inference.single_IC_inference(
        model=model,
        n_timesteps=1,
        initial_condition=IC_ds,
        device=device,
        vocal=True
    )
    tds = tendency_ds.isel(time=1) - tendency_ds.isel(time=0)
    tds.to_netcdf(IC_tendency_path)

# make heating field
km = 1.e3
amp = 0.1
k = 6
ylat = 0.
xlon = 120.
locRad = 10000.*km
heating = amp*inference.gen_elliptical_perturbation(IC_ds.latitude,IC_ds.longitude,k,ylat,xlon,locRad)

heating_ds = inference.create_empty_sfno_ds()

# find levels between 1000 and 200 hPa (inclusive)
levs = IC_ds["level"].values
levs = levs[(levs <= 1000) & (levs >= 200)]

# set perturbation temp profile to `heating` field
heating_ds["T"].loc[dict(level=levs)] = heating

# set perturbation to zero for all other variables
zero_vars = ["U", "V", "Z", "R", "VAR_10U", "VAR_10V", "VAR_100U", "VAR_100V", "SP", "MSL", "TCW", "VAR_2T"]
for var in zero_vars:
    heating_ds[var][:] = 0.
    
rpert = - tds + heating_ds
# test with below line -- model output should be static
# rpert = -tds

ds = inference.single_IC_inference(
        model=model,
        n_timesteps=n_timesteps,
        initial_condition=IC_ds,
        recurrent_perturbation=rpert,
        device=device,
        vocal=True
    )

# add some metadata
ds = ds.rename({"time": "lead_time"})
ds = ds.assign_coords({"lead_time": lead_times_h})
ds = ds.assign_attrs({"lead_time": "Lead time in hours"})

# save output
print(f"Saving output to {output_path}.")
if output_path.exists():
    print(f"Output file already exists. Overwriting.")
    output_path.unlink()
ds.to_netcdf(output_path)
