from utils import general
from torch.cuda import mem_get_info
from earth2studio.io import XarrayBackend
from earth2studio.data import CDS
import earth2studio.run as run
import xarray as xr
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
    tendency_file = output_dir / "auxiliary" / f"{model_name}_tendency.nc"
    nc_output_file = output_dir / f"{model_name}_output.nc"

    # load the model
    model = general.load_model(model_name)
    model.const_sza = True

    # ic is either a date (use CDS) or a path to a netcdf file (use Xarray)
    ic_val = config["initial_condition_params"]["ic"]
    if ic_val.startswith("dt"):
        # load the initial condition times
        ic_date = np.datetime64(ic_val[3:])
        # get ERA5 data from the ECMWF CDS
        data_source = CDS()
    elif ic_val.startswith("xr"):
        ic_file = Path(ic_val[3:]) / f"{model_name}.nc"
        IC_ds = general.sort_latitudes(xr.open_dataset(ic_file), model_name, input=True)
        ic_date = IC_ds["time"].values[-1] # use last time in file for cases where multiple times present, e.g. FuXi
        # make data into custom data source
        data_source = general.DataSet(IC_ds, model_name)

    # interface between model and data
    xr_io = XarrayBackend()
    
    # run experiment
    run_kwargs = {
        "time": np.atleast_1d(ic_date),
        "nsteps": config["n_timesteps"],
        "prognostic": model,
        "data": data_source,
        "io": xr_io,
        "device": config["device"],
    }

    ds = general.run_deterministic_w_perturbations(
        run_kwargs,
        config["tendency_reversion"],
        model_name,
        tendency_file,
    )

    # for clarity
    ds = ds.rename({"time": "init_time"})

    # only keep a few common vars for testing
    ds = ds[["msl", "t2m", "z500"]]

    # add model dimension to enable opening with open_mfdataset
    ds = ds.assign_coords(model=model_name)

    # save data
    ds.to_netcdf(nc_output_file)


if __name__ == "__main__":
    general.run_experiment_controller(
        calling_directory=Path(__file__).parent,
        run_experiment=run_experiment,
        config_path=Path(__file__).parent / "0.config.yaml"
    )
