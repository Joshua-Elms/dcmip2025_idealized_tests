import xarray as xr
import numpy as np
from pathlib import Path
import numpy as np
from torch.cuda import mem_get_info
from utils_E2S import general
from earth2studio.io import XarrayBackend
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def run_experiment(model_name: str, config_path: str) -> str:
    # read config file
    config = general.read_config(config_path)

    print(f"Running experiment for model: {model_name}")
    print(
        f"GPU memory: {mem_get_info()[0] / 1e9:.2f} GB available out of {mem_get_info()[1] / 1e9:.2f} GB"
    )

    # unpack config & set paths
    season = config["IC_season"]
    IC_path = Path(config["HM24_IC_dir"]) / f"{model_name}.nc"
    perturbation_path = Path(config["perturbation_dir"]) / f"{season}_40N_150E_z-regression_{model_name}.nc"
    output_dir = Path(config["experiment_dir"]) / config["experiment_name"]
    nc_output_file = output_dir / f"output_{model_name}.nc"
    tendency_file = output_dir / "auxiliary" / f"tendency_{model_name}.nc"

    # load the model
    model = general.load_model(model_name)
    if model_name in ["SFNO", "GraphCastOperational"]:
        model.const_sza = True

    # interface between model and data
    xr_io = XarrayBackend()

    # load the time-mean initial condition from HM24
    IC_ds = xr.open_dataset(IC_path)
    IC_ds = general.sort_latitudes(IC_ds, model_name, input=True)
    data_source = general.DataSet(IC_ds, model_name)

    # read and preprocess initial perturbation
    pert = xr.open_dataset(perturbation_path)
    amp = config["perturbation_params"]["amp"]
    pert = pert * amp

    # run experiment
    run_kwargs = {
        "time": np.atleast_1d(np.datetime64("2000-01-01")),
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
        initial_perturbation=pert,
    )

    # for clarity
    ds = ds.rename({"time": "init_time"})

    # add model dimension to enable opening with open_mfdataset
    ds = ds.assign_coords(model=model_name)

    # sort output
    ds = general.sort_latitudes(ds, model_name, input=False)

    # save data
    ds.to_netcdf(nc_output_file)
