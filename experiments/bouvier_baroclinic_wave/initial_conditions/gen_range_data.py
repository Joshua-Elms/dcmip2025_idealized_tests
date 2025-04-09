from utils import *
import numpy as np
from itertools import product
import yaml
from pathlib import Path

# read configuration
config_path = Path('/glade/u/home/jmelms/projects/dcmip2025_idealized_tests/initial_conditions/bouvier_et_al_2024/configs/template.yml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
ic_params = config["initial_condition_parameters"]
    
# set up directories
exp_dir = Path(config["experiment_dir"]) / config["experiment_name"] # all data for experiment stored here
exp_dir.mkdir(parents=True, exist_ok=True) # make dir if it doesn't exist
ic_csv_dir = exp_dir / "ic_csv" # contains fort generated ICs, must be processed into nc before used for inference
ic_csv_dir.mkdir(exist_ok=True)
ic_nc_dir = exp_dir / "ic_nc" # contains processed ICs in nc format, ready for inference
ic_nc_dir.mkdir(exist_ok=True)
output_dir = exp_dir / "output" # contains output from inference
output_dir.mkdir(exist_ok=True)

# find the iterable parameter (only one allowed currently)
keys, (vals, units) = zip(*ic_params.items())
val_types = [isinstance(v, list) for v in vals]
iter_param_idx = val_types.index(1)
iter_param = keys[iter_param_idx]
iter_param_units = units[iter_param_idx]
assert sum(val_types) == 1, "Only one iterable parameter allowed"

print(f"iterating over {iter_param}: {ic_params[iter_param]}")
for i, val in enumerate(ic_params[iter_param]):
    
    # generate csv initial condition from fort executable
    csv_ic_path = ic_csv_dir / f"ic_{iter_param}={val}.csv"
    csv_kwargs = {
        **ic_params,
        "executable_path": config["fort_executable_path"],
        iter_param: val, # must go after **ic_params to overwrite iter_param value
        "filename": csv_ic_path,
    }
    
    out, err = run_fortran_executable(**csv_kwargs)
    
    # process csv initial condition into netcdf and add some derived variables to it
    nc_ic_path = ic_nc_dir / f"ic_{iter_param}={val}.nc"
    nc_kwargs = {
        **config["processor_parameters"],
        "csv_path": csv_ic_path,
        "nc_path": nc_ic_path,
        "nlat": ic_params["nlat"],
        "metadata_dir": Path(config["processor_parameters"]["metadata_dir"]),
    }
    ds = process_individual_fort_file(**nc_kwargs)
    

print("program should end now, feel free to end via CTRL+C")