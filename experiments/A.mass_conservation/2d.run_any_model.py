import xarray as xr
import datetime as dt
import torch
# from earth2mip import networks # type: ignore
from utils_E2S import inference_sfno as inference
from utils_E2S.general import DataSet
from pathlib import Path
import yaml
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

""" Import libraries. """
import os
from dotenv import load_dotenv

load_dotenv();  # TODO: make common example prep function

from earth2studio.io import XarrayBackend
from earth2studio.data import CDS
from earth2studio.models.px import SFNO, Pangu6, GraphCastOperational, FuXi

import earth2studio.run as run
cache_loc = os.environ["EARTH2STUDIO_CACHE"]
print(f"Earth2Studio cache: {cache_loc}")

print("\n\nRunning SFNO model inference for mass conservation experiment.\n\n")

# read configuration
this_dir = Path(__file__).parent
config_path = this_dir / "0.config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# set up directories
exp_dir = Path(config["experiment_dir"]) / config["experiment_name"] # all data for experiment stored here
exp_dir.mkdir(parents=True, exist_ok=True) # make dir if it doesn't exist
output_path = exp_dir / "sfno_output.nc" # where to save output from inference

# load the model
device = config["device"]
assert device in ["cpu", "cuda", "cuda:0"], "Device must be 'cpu' or 'cuda'."
print(f"Loading model on {device}.")
""" Set up the model """
# Load the default model package which downloads the check point from NGC
package = SFNO.load_default_package()
model = SFNO.load_model(package)
package = Pangu6.load_default_package()
model = Pangu6.load_model(package)
package = GraphCastOperational.load_default_package()
model = GraphCastOperational.load_model(package)
package = FuXi.load_default_package()
model = FuXi.load_model(package)
print("Model loaded.")

# load the initial condition times
ic_dates = [dt.datetime.strptime(str_date, "%Y/%m/%d %H:%M") for str_date in config["ic_dates"]]
time = [dt.datetime(1850,1,1) + dt.timedelta(hours=i*6) for i in range(config["n_timesteps"]+1)]

# start loop
sp_da_stack = []
msl_da_stack = []
for d, date in enumerate(ic_dates):
    # set the initial time
    init_time = date
    end_time = date + dt.timedelta(hours=6*config["n_timesteps"])
    
    dummy_io = XarrayBackend()

    # generate the initial condidtion
    print(f"Running IC {d+1}/{len(ic_dates)}: {init_time}.")
    data_source = CDS()
    ds = run.deterministic(
        time=np.atleast_1d(init_time), 
        nsteps=config["n_timesteps"],
        prognostic=model,
        data=data_source,
        io=dummy_io,
        device=device,
    )
    sp_da_stack.append(ds.root["sp"])
    msl_da_stack.append(ds.root["msl"])

        
# stack the output data by init time
sp_da = xr.concat(sp_da_stack, dim="time")
sp_da = sp_da.rename({"time": "init_time"})
msl_da = xr.concat(msl_da_stack, dim="time")
msl_da = msl_da.rename({"time": "init_time"})
# create the output dataset
ds_out = xr.Dataset({
    "SP": sp_da,
    "MSL": msl_da
})

# add initialization coords
ds_out = ds_out.assign_coords({"init_time": ic_dates})

# postprocess data
ds_out["SP"] = ds_out["SP"] / 100 # convert from Pa to hPa
ds_out["MEAN_SP"] = inference.latitude_weighted_mean(ds_out["SP"], ds.root.lat)
ds_out["IC_MEAN_SP"] = ds_out["MEAN_SP"].mean(dim="init_time")

ds_out["MSL"] = ds_out["MSL"] / 100 # convert from Pa to hPa
ds_out["MEAN_MSL"] = inference.latitude_weighted_mean(ds_out["MSL"], ds.root.lat)
ds_out["IC_MEAN_MSL"] = ds_out["MEAN_MSL"].mean(dim="init_time")

# save to data dir in same directory as this file
if output_path.exists():
    print(f"File {output_path} already exists. Overwriting.")
    output_path.unlink() # delete the file if it exists
ds_out.to_netcdf(output_path)