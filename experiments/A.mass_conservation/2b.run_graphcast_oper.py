import xarray as xr
import datetime as dt
import torch
from earth2mip import networks # type: ignore
from utils import inference_graphcast_oper as inference
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

print("\n\nRunning graphcast_operational model inference for mass conservation experiment.\n\n")

# read configuration
this_dir = Path(__file__).parent
config_path = this_dir / "0.config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# set up directories
exp_dir = Path(config["experiment_dir"]) / config["experiment_name"] # all data for experiment stored here
exp_dir.mkdir(parents=True, exist_ok=True) # make dir if it doesn't exist
output_path = exp_dir / "graphcast_oper_output.nc" # where to save output from inference

# load the model
device = config["device"]
assert device in ["cpu", "cuda", "cuda:0"], "Device must be 'cpu' or 'cuda'."
print(f"Loading model on {device}.")
model = networks.get_model("e2mip://graphcast_operational",device=device)
print("Model loaded.")

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
    print(f"Running IC {d+1}/{len(ic_dates)}: {init_time}.")
    x = inference.rda_era5_to_graphcast_oper_state(device=device, time = init_time)

    # run the model
    lead_times = []
    data_list = []
    iterator = model(init_time, x)
    for k, (time, data, _) in enumerate(iterator):
        print(f"Step {time} completed.")

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
    ds = inference.unpack_graphcast_oper_state(data, time = lead_times)
    ds = ds.assign_coords({"lead_time": lead_times})
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
    print(f"File {output_path} already exists. Overwriting.")
    output_path.unlink() # delete the file if it exists
ds_out.to_netcdf(output_path)