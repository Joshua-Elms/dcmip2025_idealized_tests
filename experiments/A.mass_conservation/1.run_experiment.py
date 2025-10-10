from utils import general, model_info
from torch.cuda import mem_get_info
import xarray as xr
from earth2studio.io import XarrayBackend, NetCDF4Backend
from earth2studio.data import CDS
import earth2studio.run as run
import numpy as np
from pathlib import Path
import datetime as dt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def run_experiment(model_name: str, config_path: str) -> str:
    # read config file
    config = general.read_config(config_path)

    print(f"Running experiment for model: {model_name}")
    print(
        f"GPU memory: {mem_get_info()[0] / 1e9:.2f} GB available out of {mem_get_info()[1] / 1e9:.2f} GB"
    )

    # set output paths
    output_dir = Path(config["experiment_dir"]) / config["experiment_name"]
    nc_output_file = output_dir / f"{model_name}_output.nc"

    # load the model
    model = general.load_model(model_name)

    # load the initial condition times
    ic_dates = [
        dt.datetime.strptime(str_date, "%Y-%m-%dT%Hz")
        for str_date in config["ic_dates"]
    ]
    tmp_output_files = [output_dir / f"{model_name}_output_{ic_date.strftime('%Y%m%dT%H')}_tmp.nc" for ic_date in ic_dates]
    
    # figure out which variables to keep, given that this model may not output 
    # all variables requested in config
    model_vars = model_info.MODEL_VARIABLES[model_name]["names"]
    keep_vars = [var for var in config["keep_vars"] if var in model_vars + ["ssp"]]
    sl_temp = "t1000" in keep_vars
    req_vars = ["msl", "z1000", "t2m", "t1000" if sl_temp else "t850"]

    ds_list = []
    
    # have to iterate like this to work w/ GraphCastOperational
    for i, ic_date in enumerate(ic_dates):
        
        fpath = tmp_output_files[i]

        # interface between model and data
        io = NetCDF4Backend(fpath)

        # get ERA5 data from the ECMWF CDS
        data_source = CDS(verbose=False)

        # run the model for all initial conditions at once
        run.deterministic(
            time=np.atleast_1d(ic_date),
            nsteps=config["n_timesteps"],
            prognostic=model,
            data=data_source,
            io=io,
            device=config["device"],
        )
        
        extract_vars = list(set(keep_vars + req_vars) - {"ssp"})
        tmp_ds = xr.open_dataset(fpath)[extract_vars]
        
        # add synthetic SP variable if requested
        if "ssp" in config["keep_vars"]:
            assert all(var in tmp_ds.data_vars for var in req_vars), f"Missing required variables for ssp calculation: {req_vars}"
            Zs = xr.open_dataset(config["surface_geopotential_path"])["geopotential"]
            g = 9.80665  # m/s^2
            zs = Zs / g  # convert to height in m
            zs = zs.rename({"longitude": "lon", "latitude": "lat"})
            z1000 = tmp_ds["z1000"] / g # convert to height in m
            Rd = 287.05  # J/(kg*K)
            T2M = tmp_ds["t2m"] # K
            T0 = tmp_ds["t1000"] # and we don't have t1013 so must use t1000 for T at MSLP level
            p0 = tmp_ds["msl"]  # in Pa
            lapse_rate = 0.0065 # K/m
            exponent = g / (Rd * lapse_rate)
            ssp = p0 * (T2M / T0) ** exponent
            tmp_ds["ssp"] = ssp
            
        tmp_ds = tmp_ds[keep_vars]
        
        # sort by latitude
        tmp_ds = general.sort_latitudes(tmp_ds, model_name, input=False)
    
        # postprocess data
        for var in keep_vars:
            tmp_ds[f"MEAN_{var}"] = general.latitude_weighted_mean(tmp_ds[var], tmp_ds.lat)
            tmp_ds[f"IC_MEAN_{var}"] = tmp_ds[f"MEAN_{var}"].mean(dim="time")
            
        # if storage is tight, drop the full 3D fields
        if config.get("keep_base_fields", True) is False:
            tmp_ds = tmp_ds.drop_vars(keep_vars)
            
        # overwrite temporary file
        fpath.unlink()
        tmp_ds.to_netcdf(fpath, mode="w")

    # combine all temporary files into one dataset, not using openmf because it's slow
    ds_list = [xr.open_dataset(file) for file in tmp_output_files]
    ds = xr.concat(ds_list, dim="time")
    for tmp_file in tmp_output_files:
        tmp_file.unlink()  # delete temporary file
    print(f"Combined dataset has dimensions: {ds.dims}")

    # add model dimension to enable opening with open_mfdataset
    ds = ds.assign_coords(model=model_name)
    
    # for clarity
    ds = ds.rename({"time": "init_time"})

    # save data
    ds.to_netcdf(nc_output_file)


if __name__ == "__main__":
    general.run_experiment_controller(
        calling_directory=Path(__file__).parent,
        run_experiment=run_experiment,
        config_path=Path(__file__).parent / "0.config.yaml"
    )
