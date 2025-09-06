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
    IC_path = Path(config["HM24_IC_dir"]) / f"{model_name}.nc"
    output_dir = Path(config["experiment_dir"]) / config["experiment_name"]
    nc_output_file = output_dir / f"output_{model_name}.nc"
    tendency_file = output_dir / "auxiliary" / f"tendency_{model_name}.nc"
    recurrent_perturbation_file = output_dir / "auxiliary" / f"heating_{model_name}.nc"

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

    # create recurrent perturbation
    pert_params = config["perturbation_params"]
    amp = pert_params["amp"]
    k = pert_params["k"]
    ylat = pert_params["ylat"]
    xlon = pert_params["xlon"]
    locRad = pert_params["locRadkm"] * 1.0e3  # convert km to m
    heating = amp * general.gen_elliptical_perturbation(
        IC_ds.lat, IC_ds.lon, k, ylat, xlon, locRad
    )
    heating = heating[np.newaxis, :]  # init_time
    # for any model with multiple input timesteps
    # we should only perturb the final one in the output
    heating_ds = general.create_initial_condition(model)
    model_levels = general.model_levels[model_name]
    model_coords = {
        k: v for k, v in model.input_coords().items() if k in ["lat", "lon"]
    }
    model_coords["time"] = np.atleast_1d(
        np.datetime64("2000-01-01")
    )  # add time coordinate
    perturb_levels = model_levels[(model_levels <= 1000) & (model_levels >= 200)]
    perturb_variables = [f"t{lev}" for lev in perturb_levels]
    for var in perturb_variables:
        heating_ds[var] = xr.DataArray(
            heating, model_coords, dims=["time", "lat", "lon"]
        )
    print(f"Applying tropical heating perturbation: {heating_ds}")
    heating_ds.to_netcdf(recurrent_perturbation_file)

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
        recurrent_perturbation=heating_ds,
    )

    # for clarity
    ds = ds.rename({"time": "init_time"})

    # add model dimension to enable opening with open_mfdataset
    ds = ds.assign_coords(model=model_name)

    # sort output
    ds = general.sort_latitudes(ds, model_name, input=False)

    # save data
    ds.to_netcdf(nc_output_file)


if __name__ == "__main__":
    general.run_experiment_controller(
        experiment_func=run_experiment,
        config_path = Path(__file__).parent / "0.config.yaml"
    )
