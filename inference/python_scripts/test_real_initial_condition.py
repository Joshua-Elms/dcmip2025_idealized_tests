import xarray as xr
import datetime as dt
import torch
import numpy as np
import logging
from earth2mip import networks
import inference.dcmip2025_helper_funcs as dcmip
import dotenv

# load the earth2mip environment variables
dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)

# initial time
init_time = dt.datetime(2017, 8, 24, 00)
end_time = dt.datetime(2017, 8, 29, 00)

# load the model
device = "cuda:0"
print("Loading model.")
model = networks.get_model("fcnv2_sm").to(device)
print("Model loaded.")

# generate the initial condidtion
print("Initializing model.")
x = dcmip.rda_era5_to_sfno_state(device=device, time = init_time)
print("Model initialized.")

# run the model
data_list = []
times = []
iterator = model(init_time, x)
for k, (time, data, _) in enumerate(iterator):
    print(f"Step {time} completed.")

    # append the data to the list
    # (move the data to the cpu (memory))
    data_list.append(data.cpu())
    # append the times too
    times.append(time)

    # check if we're at the end time
    if time >= end_time:
        break

# stack the output data by time
data = torch.stack(data_list)

# unpack the data into an xarray object
ds = dcmip.unpack_sfno_state(data, time = times)

# save the data
ds.squeeze().to_netcdf("harvey_test.nc")
