import xarray as xr
from pathlib import Path
import numpy as np
import warnings
import datetime as dt
from torch.cuda import mem_get_info
from utils_E2S import general
from earth2studio.io import XarrayBackend
from earth2studio.data import CDS
import earth2studio.run as run
warnings.filterwarnings("ignore", category=FutureWarning)

# ### Setup Specific Experiment ###

# depending on the experiment, grab appropriate parameters & prepare perturbation
# if hm24_exp_name == "tropical_heating":
#     heating_ds_path = exp_dir / "heating.nc"
#     pert_params = config["perturbation_params"]
#     amp = pert_params["amp"]
#     k = pert_params["k"]
#     ylat = pert_params["ylat"]
#     xlon = pert_params["xlon"]
#     locRad = pert_params["locRadkm"] * 1.e3 # convert km to m
    
#     # make heating field
#     heating = amp*inference.gen_elliptical_perturbation(IC_ds.latitude,IC_ds.longitude,k,ylat,xlon,locRad)
#     heating_ds = inference.create_empty_sfno_ds()
    
#     # find levels between 1000 and 200 hPa (inclusive)
#     levs = IC_ds["level"].values
#     levs = levs[(levs <= 1000) & (levs >= 200)]

#     # set perturbation temp profile to `heating` field
#     heating_ds["T"].loc[dict(level=levs)] = heating

#     # set perturbation to zero for all other variables
#     zero_vars = ["U", "V", "Z", "R", "VAR_10U", "VAR_10V", "VAR_100U", "VAR_100V", "SP", "MSL", "TCW", "VAR_2T"]
#     for var in zero_vars:
#         heating_ds[var][:] = 0.
        
#     # save heating_ds to file
#     print(f"Saving heating dataset to {heating_ds_path}.")
#     print(f"Applying tropical heating perturbation: {heating_ds}")
#     heating_ds.to_netcdf(heating_ds_path)
    
#     # these two get passed to inference.single_IC_inference (via rpert for f)
#     initial_perturbation = None
#     f = heating_ds
    
# # initial_perturbation might be a smaller region than IC_ds
# # so we add to an empty dataset with the same shape as IC_ds
# if initial_perturbation is not None:
#     zero_ds = IC_ds.copy(deep=True) * 0  # create a zero dataset with the same shape as IC_ds
    
#     lat_coords = initial_perturbation.latitude.values
#     lon_coords = initial_perturbation.longitude.values
    
#     # Create a selection dictionary with the coordinates from the smaller dataset
#     selection = {
#         'latitude': lat_coords,
#         'longitude': lon_coords
#     }
    
#     # Add the smaller array to the larger array at the selected coordinates
#     zero_ds.loc[selection] += initial_perturbation
    
#     initial_perturbation = zero_ds  # now initial_perturbation has the same shape as IC_ds
#     assert initial_perturbation.dims == IC_ds.dims, \
#         f"Initial perturbation dimensions {initial_perturbation.dims} do not match IC dimensions {IC_ds.dims}. Check the perturbation shape."
    


def run_experiment(model_name: str, config_path: str) -> str:
    # read config file
    config = general.read_config(config_path)
    
    print(f"Running experiment for model: {model_name}")
    print(f"GPU memory: {mem_get_info()[0] / 1e9:.2f} GB available out of {mem_get_info()[1] / 1e9:.2f} GB")
    
    # unpack config & set paths
    IC_path = Path(config["HM24_IC_dir"]) / f"{model_name}.nc"
    output_dir = Path(config["experiment_dir"]) / config["experiment_name"]
    nc_output_file = output_dir / f"output_{model_name}.nc"
    tendency_file = output_dir  / "auxiliary" / f"tendency_{model_name}.nc"
    recurrent_perturbation_file = output_dir / "auxiliary" / f"heating_{model_name}.nc"

    # load the model
    model = general.load_model(model_name)
    if model_name == "SFNO":
        model.const_sza = True

    # interface between model and data
    xr_io = XarrayBackend()
    
    # load the time-mean initial condition from HM24
    IC_ds = xr.open_dataset(IC_path)
    IC_ds = general.sort_latitudes(IC_ds, model_name, input=True)

    # get ERA5 data from the ECMWF CDS
    data_source = general.DataSet(
        IC_ds,
        model_name
    )
    
    # create recurrent perturbation
    pert_params = config["perturbation_params"]
    amp = pert_params["amp"]
    k = pert_params["k"]
    ylat = pert_params["ylat"]
    xlon = pert_params["xlon"]
    locRad = pert_params["locRadkm"] * 1.e3 # convert km to m
    heating = amp * general.gen_elliptical_perturbation(IC_ds.lat,IC_ds.lon,k,ylat,xlon,locRad)
    heating = heating[np.newaxis, :]  # init_time and lead_time
    heating_ds = general.create_initial_condition(model).squeeze("lead_time")
    model_levels = general.model_levels[model_name]
    model_coords = {k:v for k, v in model.input_coords().items() if k in ["lat", "lon"]}
    model_coords["time"] = np.atleast_1d(np.datetime64("2000-01-01"))  # add time coordinate
    perturb_levels = model_levels[(model_levels <= 1000) & (model_levels >= 200)]
    perturb_variables = [f"t{lev}" for lev in perturb_levels]
    for var in perturb_variables:
        heating_ds[var] = xr.DataArray(heating, model_coords, dims=["time", "lat", "lon"])
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
        recurrent_perturbation=heating_ds
        )
    
    # for clarity
    ds = ds.rename({"time": "init_time"}) 
        
    # add model dimension to enable opening with open_mfdataset
    ds = ds.assign_coords(model=model_name)
    
    # sort output
    ds = general.sort_latitudes(ds, model_name, input=False)

    # save data
    ds.to_netcdf(nc_output_file)