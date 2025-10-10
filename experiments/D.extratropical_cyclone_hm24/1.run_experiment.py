from utils import general
from torch.cuda import mem_get_info
from earth2studio.io import XarrayBackend
import xarray as xr
import numpy as np
from pathlib import Path
import metpy
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
        / f"{season}_40N_150E_z-regression_{model_name}.nc"
    )
    output_dir = Path(config["experiment_dir"]) / config["experiment_name"]
    nc_output_file = output_dir / f"output_{model_name}.nc"
    tendency_file = output_dir / "auxiliary" / f"tendency_{model_name}.nc"

    # load the model
    model = general.load_model(model_name)
    
    # some models (SFNO, FCN3, ...) need to be told to hold the solar zenith angle constant
    model.const_sza = True

    # interface between model and data
    xr_io = XarrayBackend()

    # load the time-mean initial condition from HM24
    IC_ds = xr.open_dataset(IC_path)
    IC_ds = general.sort_latitudes(IC_ds, model_name, input=True)
    data_source = general.DataSet(IC_ds, model_name)

    # read and preprocess initial perturbation
    pert = xr.open_dataset(perturbation_path)
    pert = general.sort_latitudes(pert, model_name, input=True)
    amp = pert_params["amp"]
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
    
    # if not present in output, derive specific humidity from relative humidity
    r_levels = [var for var in ds.data_vars if var.startswith("r") and var[1:].isdigit()]
    q_levels = [var.replace("r", "q") for var in r_levels]
    
    for r_var, q_var in zip(r_levels, q_levels):
        if q_var in ds.data_vars or f"t{r_var[1:]}" not in ds.data_vars:
            # Skip if specific humidity already present or if temperature at that level is missing
            continue
        print(f"Deriving {q_var} from {r_var}")
        plev = int(r_var[1:])
        plev_hPa = plev * metpy.units.units("hPa")
        q = metpy.calc.specific_humidity_from_dewpoint(
            pressure=plev_hPa,
            dewpoint=metpy.calc.dewpoint_from_relative_humidity(
                temperature=ds["t" + r_var[1:]] * metpy.units.units.kelvin,
                relative_humidity=(ds[r_var]/100) * metpy.units.units.percent, 
            ),
            phase="auto", # set phase depending on temperature
        )
        ds[q_var] = q  # convert to kg/kg and drop units
    # save data
    ds.to_netcdf(nc_output_file)


if __name__ == "__main__":
    general.run_experiment_controller(
        calling_directory=Path(__file__).parent,
        run_experiment=run_experiment,
        config_path=Path(__file__).parent / "0.config.yaml"
    )
