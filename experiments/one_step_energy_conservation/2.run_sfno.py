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
data_dir.mkdir(parents=True, exist_ok=True) # make dir if it doesn't exist

# read configuration
config_path = this_dir / "0.config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# load the model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}.")
model = networks.get_model("fcnv2_sm").to(device)
print("Model loaded.")

# load the initial condition times
ic_dates = [dt.datetime.strptime(str_date, "%Y/%m/%d %H:%M") for str_date in config["ic_dates"]]
lead_times_h = np.arange(0, 6*config["n_timesteps"]+1, 6)
delta_Ts = config["temp_deltas_degrees_kelvin"]

# load the earth2mip environment variables
dotenv.load_dotenv()

# start loop
ds_stack_outer = []
for d, date in enumerate(ic_dates):
    ds_stack_inner = []
    for T in delta_Ts:
        delta_T_ds = inference.create_empty_sfno_ds()

        # set perturbation temp profile to T
        delta_T_ds["T"][:] = T
        delta_T_ds["VAR_2T"][:] = T
        
        # set perturbation to zero for all other variables
        zero_vars = ["U", "V", "Z", "R", "VAR_10U", "VAR_10V", "VAR_100U", "VAR_100V", "SP", "MSL", "TCW"]
        for var in zero_vars:
            delta_T_ds[var][:] = 0.
        
        ds = inference.single_IC_inference(
            model=model,
            n_timesteps=config["n_timesteps"],
            init_time=date,
            perturbation=delta_T_ds,
            device=device,
            vocal=True
        )
        ds_stack_inner.append(ds)
        
    # stack all members of the same init time together
    ds_inner = xr.concat(ds_stack_inner, dim="delta_T")
    ds_inner["delta_T"] = delta_Ts
    ds_inner["init_time"] = date
    ds_inner = ds_inner.assign_coords({"init_time": date})
    ds_stack_outer.append(ds_inner)

# stack the output data by init time
ds_out = xr.concat(ds_stack_outer, dim="init_time")

# add initialization coords
ds_out = ds_out.assign_coords({"init_time": ic_dates})

# it's a timedelta, so we're going to rename the time coordinate to lead_time
ds_out = ds_out.rename({"time": "lead_time"})

# let's keep only the variables needed for analysis
necessary_vars = ["SP", "MSL", "T", "Z", "U", "V", "TCW", "VAR_2T"]
# here you could add keep_vars from the config w/ simple list concatenation
ds_out = ds_out[necessary_vars]

# save to data dir in same directory as this file
save_path = Path(__file__).parent / "data" / "output.nc"
if save_path.exists():
    print(f"File {save_path} already exists. Deleting.")
    save_path.unlink() # delete the file if it exists
ds_out.to_netcdf(save_path)