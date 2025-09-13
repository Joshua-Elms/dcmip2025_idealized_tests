from utils import general, model_info
from torch.cuda import mem_get_info
from earth2studio.io import XarrayBackend
import xarray as xr
import numpy as np
from pathlib import Path
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
    IC_params = config["initial_condition_params"]
    season = IC_params["season"]
    IC_path = Path(IC_params["HM24_IC_dir"]) / f"{model_name}.nc"
    pert_params = config["perturbation_params"]
    perturbation_path = (
        Path(pert_params["perturbation_dir"])
        / f"{season}_15N_320E_z-regression_{model_name}.nc"
    )
    output_dir = Path(config["experiment_dir"]) / config["experiment_name"]
    nc_output_file = output_dir / f"output_{model_name}.nc"
    tendency_file = output_dir / "auxiliary" / f"tendency_{model_name}.nc"

    # load the model
    model = general.load_model(model_name)
    
    # some models (SFNO, FCN3, ...) need to be told to hold the solar zenith angle constant
    model.const_sza = True
    
    # figure out which variables to keep, given that this model may not output 
    # all variables requested in config
    model_vars = model_info.MODEL_VARIABLES[model_name]["names"]
    keep_vars = [var for var in config["keep_vars"] if var in model_vars]
    
    # load the time-mean initial condition from HM24
    IC_ds = xr.open_dataset(IC_path)
    IC_ds = general.sort_latitudes(IC_ds, model_name, input=True)

    # read and preprocess initial perturbation
    pert = xr.open_dataset(perturbation_path)
    pert = general.sort_latitudes(pert, model_name, input=True)
    amp_vec = pert_params["amp_vec"]
    
    ds_list = []
    for i, amp in enumerate(amp_vec):
        breakpoint()
        # run experiment in loop, applying diff pert amplitude each iteration
        run_kwargs = {
            "time": np.atleast_1d(np.datetime64("2000-01-01")),
            "nsteps": config["n_timesteps"],
            "prognostic": model,
            "data": general.DataSet(IC_ds, model_name),
            "io": XarrayBackend(),
            "device": config["device"],
        }
        print(f"Running with perturbation amplitude {amp} ({i+1} of {len(amp_vec)})")
        pert_scaled = pert.copy() * amp

        ds = general.run_deterministic_w_perturbations(
            run_kwargs,
            config["tendency_reversion"],
            model_name,
            tendency_file,
            initial_perturbation=pert_scaled,
        )[keep_vars]
        
        # add amplitude dimension
        ds = ds.expand_dims({"amplitude": [amp]})

        ds_list.append(ds)

    # combine all amplitudes into one dataset
    ds = xr.concat(ds_list, dim="amplitude")
    
    # sort output
    ds = general.sort_latitudes(ds, model_name, input=False)
    
    # for clarity
    ds = ds.rename({"time": "init_time"})
    
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
