import xarray as xr
import datetime as dt
import torch
import logging
from earth2mip import networks # type: ignore
from utils import inference_graphcast_small as inference
from pathlib import Path
import numpy as np
import yaml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# read configuration
this_dir = Path(__file__).parent
config_path = this_dir / "0.config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# set up directories
exp_dir = Path(config["experiment_dir"]) / config["experiment_name"] # all data for experiment stored here
exp_dir.mkdir(parents=True, exist_ok=True) # make dir if it doesn't exist
output_path = exp_dir / "graphcast_output.nc" # where to save output from inference
log_path = exp_dir / "graphcast.log" # where to save log

# set up logging
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s:%(message)s',
    datefmt='%Y-%m-%dH%H:%M:%S'
)

# load the model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.info(f"Loading model on {device}.")
model = networks.get_model("e2mip://graphcast_small",device=device)
logging.info("Model loaded.")

# load the initial condition times
ic_dates = [dt.datetime.strptime(str_date, "%Y/%m/%d %H:%M") for str_date in config["ic_dates"]]
time = [dt.datetime(1850,1,1) + dt.timedelta(hours=i*6) for i in range(config["n_timesteps"]+1)]

# start loop
da_stack = []
for d, date in enumerate(ic_dates):
    # set the initial time
    init_time = date
    end_time = date + dt.timedelta(hours=6*config["n_timesteps"])

    # generate the initial condidtion
    logging.info(f"Initializing model {d}: {init_time}.")
    x = inference.rda_era5_to_graphcast_state(device=device, time = init_time)

    # run the model
    lead_times = []
    data_list = []
    iterator = model(init_time, x)
    for k, (time, data, _) in enumerate(iterator):
        logging.info(f"Step {time} completed.")

        # append the data to the list
        # (move the data to the cpu (memory))
        data_list.append(data.cpu())
        lead_times.append(k*6) # 6 hour lead time increments

        # check if we're at the end time
        if time >= end_time:
            break

    # stack the output data by time
    data = torch.stack(data_list)

    # unpack the data into an xarray object
    ds = inference.unpack_graphcast_state(data, time = lead_times)
    da_stack.append(ds["MSL"])

        
# stack the output data by init time
da_out = xr.concat(da_stack, dim="init_time")
ds_out = da_out.to_dataset(name="MSL")

# add initialization coords
ds_out = ds_out.assign_coords({"init_time": ic_dates})

# postprocess data
ds_out["MSL"] = ds_out["MSL"] / 100 # convert from Pa to hPa
ds_out["MEAN_MSL"] = inference.latitude_weighted_mean(ds_out["MSL"], ds.latitude)
ds_out["IC_MEAN_MSL"] = ds_out["MEAN_MSL"].mean(dim="init_time")

# save to data dir in same directory as this file
if output_path.exists():
    logging.info(f"File {output_path} already exists. Overwriting.")
    output_path.unlink() # delete the file if it exists
ds_out.to_netcdf(output_path)