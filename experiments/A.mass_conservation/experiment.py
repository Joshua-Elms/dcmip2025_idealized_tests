import datetime as dt
from torch.cuda import mem_get_info
from utils_E2S import general
from pathlib import Path
import numpy as np
from earth2studio.io import XarrayBackend
from earth2studio.data import CDS
import earth2studio.run as run


def run_experiment(model_name: str, config_path: str) -> str:
    # read config file
    config = general.read_config(config_path)
    
    print(f"Running experiment for model: {model_name}")
    print(f"GPU memory: {mem_get_info()[0] / 1e9:.2f} GB available out of {mem_get_info()[1] / 1e9:.2f} GB")
    
    # set output paths
    output_dir = Path(config["experiment_dir"]) / config["experiment_name"]
    nc_output_file = output_dir / f"{model_name}_output.nc"

    # load the model
    model = general.load_model(model_name)

    # load the initial condition times
    ic_dates = [dt.datetime.strptime(str_date, "%Y/%m/%d %H:%M") for str_date in config["ic_dates"]]

    # interface between model and data
    xr_io = XarrayBackend()

    # get ERA5 data from the ECMWF CDS
    data_source = CDS() 

    # run the model for all initial conditions at once
    ds = run.deterministic(
        time=np.atleast_1d(ic_dates), 
        nsteps=config["n_timesteps"],
        prognostic=model,
        data=data_source,
        io=xr_io,
        device=config["device"],
    ).root

    # for clarity
    ds = ds.rename({"time": "init_time"}) 

    # only keep surface pressure variables
    keep_vars = config["keep_vars_dict"][model_name]
    ds = ds[keep_vars]

    # postprocess data
    for var in keep_vars:
        ds[f"MEAN_{var}"] = general.latitude_weighted_mean(ds[var], ds.lat)
        ds[f"IC_MEAN_{var}"] = ds[f"MEAN_{var}"].mean(dim="init_time")
        
    # add model dimension to enable opening with open_mfdataset
    ds = ds.assign_coords(model=model_name)

    # save data
    ds.to_netcdf(nc_output_file)