import xarray as xr
import datetime as dt
import torch
import numpy as np
import logging
from earth2mip import networks
import inference.dcmip2025_helper_funcs as dcmip
import dotenv
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# load the earth2mip environment variables
dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)

save_output = True
n_steps = 10
mid = 288 # K
half_range = 20 # K
step = 5 # K
zt0_range = np.arange(mid-half_range, mid+half_range+step, step)
var = "zt0"
output_path = f"/glade/work/jmelms/data/dcmip2025_idealized_tests/initial_conditions/steady-state/output/nt={n_steps}_ne={len(zt0_range)}_var={var}.nc"


# load the model
device = "cuda:0"
#device = "cpu"
print("Loading model.")
model = networks.get_model("fcnv2_sm").to(device)
print("Model loaded.")

# load existing initial condition
ic_dir = Path("/glade/work/jmelms/data/dcmip2025_idealized_tests/initial_conditions/steady-state/ics")
assert ic_dir.exists(), "Initial condition dir given does not exist"
ic_paths = [path for path in sorted(ic_dir.glob("*")) if int(path.stem.split("_")[-1]) in zt0_range]

for i, ic_path in enumerate(ic_paths):
    
    zt0 = int(ic_path.stem.split("_")[-1])
    ds_in = xr.open_dataset(ic_path)
    # put the initial condition into a format compatible with the SFNO
    x = dcmip.pack_sfno_state(ds_in, device=device)

    print(f"Loaded IC-{i}: {zt0 = }")

    # run the model
    data_list = []
    iterator = model(dt.datetime(1850,1,1), x)
    for k, (time, data, _) in enumerate(iterator):
        print(f"Step {k+1} completed.")
        data_list.append(data.to(device="cpu"))
        if k == n_steps:
            break

    # stack the output data by time
    data = torch.stack(data_list)

    # unpack the data into an xarray object
    ds = dcmip.unpack_sfno_state(data)
    if i == 0:
        ds_out = ds
        
    else:
        ds_out = xr.concat([ds_out, ds], dim="ensemble")

ds_out.assign_coords({"ensemble": zt0_range})       
ds_out.to_netcdf(output_path)

print("Output: ")

print(ds_out)

print(f"Saved ds to {output_path}")