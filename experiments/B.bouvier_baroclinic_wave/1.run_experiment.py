from utils import general, model_info
import initial_conditions.Bouvier.data_utils as IC_utils
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
    output_dir = Path(config["experiment_dir"]) / config["experiment_name"]
    nc_output_file = output_dir / f"output_{model_name}.nc"
    tendency_file = output_dir / "auxiliary" / f"tendency_{model_name}.nc"

    # load the model
    model = general.load_model(model_name)

    # some models (SFNO, FCN3, ...) need to be told to hold the solar zenith angle constant
    model.const_sza = True

    # interface between model and data
    xr_io = XarrayBackend()

    # gen initial conditions step 1: run fortran executable to generate csv
    csv_path = (
        output_dir / "auxiliary" / "fort_output.csv"
    ).as_posix()
    if not Path(csv_path).exists():
        config["initial_condition_params"]["output_filename"] = csv_path
        out, err = IC_utils.run_fortran_executable(**config["initial_condition_params"])
        print(f"Fortran executable output: {out}")
        if err:
            print(f"Fortran executable error: {err}")
        
    # gen initial conditions step 2: process csv into xarray dataset
    super_IC_path = (
        output_dir / "auxiliary" / f"initial_conditions_all_vars.nc"
    )
    if not super_IC_path.exists():
        IC_utils.generate_superset_of_initial_conditions(
            csv_path=Path(csv_path),
            nc_path=super_IC_path,
            **config["processor_params"],
        )
    super_IC_ds = xr.open_dataset(super_IC_path)
    IC_ds = super_IC_ds[model_info.MODEL_VARIABLES[model_name]["names"]]   
    print(f"Grabbed subset of variables from super IC dataset.")

    # prepare data source
    IC_ds = general.sort_latitudes(IC_ds, model_name, input=True)
    data_source = general.DataSet(IC_ds, model_name)

    # generate perturbation
    pert_params = config["perturbation_params"]
    if pert_params["enabled"]:
        ylat = pert_params["center_latitude"]
        xlon = pert_params["center_longitude"]
        locRad = pert_params["localization_radius"]
        u_pert_base = pert_params["u_wind_pert_base"]
        
        # set up the perturbation
        upert = xr.DataArray(
            general.gen_baroclinic_wave_perturbation(
                IC_ds.lat, IC_ds.lon, ylat, xlon, u_pert_base, locRad
            ),
            coords=[IC_ds.lat.values, IC_ds.lon.values],
            dims=["lat", "lon"],
        )
        upert = general.sort_latitudes(upert, model_name, input=True)
        pert = general.create_initial_condition(model)
        
        # if model uses 720 lats instead of 721, slice upert to match
        if pert.sizes["lat"] == 720:
            # TODO: figure out consistent way to handle this across models;
            # for now just slice off the first latitude
            upert = upert[1:]

        # set perturbation u-wind profile to `upert` field
        u_vars = [var for var in model_info.MODEL_VARIABLES[model_name]["names"] if var.startswith("u")]
        for u_var in u_vars:
            pert[u_var] = upert

        pert.to_netcdf(
            output_dir / "auxiliary" / f"initial_perturbation_{model_name}.nc"
        )

    else:
        pert = None

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


if __name__ == "__main__":
    general.run_experiment_controller(
        calling_directory=Path(__file__).parent,
        run_experiment=run_experiment,
        config_path=Path(__file__).parent / "0.config.yaml",
    )
