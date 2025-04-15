import xarray as xr
import datetime as dt
import torch
import numpy as np
import logging
from earth2mip import networks # type: ignore
import utils.data as dcmip
import dotenv
from pathlib import Path
import numpy as np
import yaml
from time import perf_counter
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
times = np.arange(0, 6*config["n_timesteps"]+1, 6)

# load the earth2mip environment variables
dotenv.load_dotenv()

# start loop
ds_stack = []
for d, date in enumerate(ic_dates):
    # set the initial time
    init_time = date
    end_time = date + dt.timedelta(hours=6*config["n_timesteps"])

    # generate the initial condidtion
    print(f"Initializing model {d}: {init_time}.")
    x = dcmip.rda_era5_to_sfno_state(device=device, time = init_time)

    # run the model
    data_list = []
    iterator = model(init_time, x)
    for k, (time, data, _) in enumerate(iterator):
        print(f"Step {time} completed.")

        # append the data to the list
        # (move the data to the cpu (memory))
        data_list.append(data.cpu())

        # check if we're at the end time
        if time >= end_time:
            break

    # stack the output data by time
    data = torch.stack(data_list)

    # unpack the data into an xarray object
    ds = dcmip.unpack_sfno_state(data, time = times)
    ds_stack.append(ds)

        
# stack the output data by init time
ds_out = xr.concat(ds_stack, dim="init_time")

# add initialization coords
ds_out = ds_out.assign_coords({"init_time": ic_dates})

# save to data dir in same directory as this file
save_path = Path(__file__).parent / "data" / "raw_output.nc"
save_path.unlink() # delete the file if it exists
ds_out.to_netcdf(save_path)