import xarray as xr
import numpy as np
from utils import inference_sfno as inference
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

    
# load initial condition
IC_dir = Path(config["time_mean_IC_dir"])
IC_season = config["IC_season"]
IC_path = IC_dir / f"{IC_season}_ERA5_time_mean_sfno.nc"
IC_ds = xr.open_dataset(IC_path).sortby("latitude", ascending=False)

### Setup Specific Experiment ###

# depending on the experiment, grab appropriate parameters & prepare perturbation
if hm24_exp_name == "tropical_heating":
    heating_ds_path = exp_dir / "heating.nc"
    pert_params = config["perturbation_params"]
    amp = pert_params["amp"]
    k = pert_params["k"]
    ylat = pert_params["ylat"]
    xlon = pert_params["xlon"]
    locRad = pert_params["locRadkm"] * 1.e3 # convert km to m
    
    # make heating field
    heating = amp*inference.gen_elliptical_perturbation(IC_ds.latitude,IC_ds.longitude,k,ylat,xlon,locRad)
    heating_ds = inference.create_empty_sfno_ds()
    
    # find levels between 1000 and 200 hPa (inclusive)
    levs = IC_ds["level"].values
    levs = levs[(levs <= 1000) & (levs >= 200)]

    # set perturbation temp profile to `heating` field
    heating_ds["T"].loc[dict(level=levs)] = heating

    # set perturbation to zero for all other variables
    zero_vars = ["U", "V", "Z", "R", "VAR_10U", "VAR_10V", "VAR_100U", "VAR_100V", "SP", "MSL", "TCW", "VAR_2T"]
    for var in zero_vars:
        heating_ds[var][:] = 0.
        
    # save heating_ds to file
    print(f"Saving heating dataset to {heating_ds_path}.")
    print(f"Applying tropical heating perturbation: {heating_ds}")
    heating_ds.to_netcdf(heating_ds_path)
    
    # these two get passed to inference.single_IC_inference (via rpert for f)
    initial_perturbation = None
    f = heating_ds
    
# initial_perturbation might be a smaller region than IC_ds
# so we add to an empty dataset with the same shape as IC_ds
if initial_perturbation is not None:
    zero_ds = IC_ds.copy(deep=True) * 0  # create a zero dataset with the same shape as IC_ds
    
    lat_coords = initial_perturbation.latitude.values
    lon_coords = initial_perturbation.longitude.values
    
    # Create a selection dictionary with the coordinates from the smaller dataset
    selection = {
        'latitude': lat_coords,
        'longitude': lon_coords
    }
    
    # Add the smaller array to the larger array at the selected coordinates
    zero_ds.loc[selection] += initial_perturbation
    
    initial_perturbation = zero_ds  # now initial_perturbation has the same shape as IC_ds
    assert initial_perturbation.dims == IC_ds.dims, \
        f"Initial perturbation dimensions {initial_perturbation.dims} do not match IC dimensions {IC_ds.dims}. Check the perturbation shape."
    
### Compute Tendencies ###

# only need to compute tendencies if tendency reversion is enabled
if tendency_reversion:
    # if file exists at tendency_path, load it
    tendency_path = tendency_dir / f"{IC_season}_sfno_tendency.nc"
    if tendency_path.exists():
        print(f"Loading cached tendencies from {tendency_path}.")
        tds = xr.open_dataset(tendency_path).sortby("latitude", ascending=False)
    # else, compute tendencies and save to tendency_path for future use
    else:
        print(f"Computing tendencies and saving to {tendency_path}.")
        tendency_ds = inference.single_IC_inference(
            model=model,
            n_timesteps=1,
            initial_condition=IC_ds,
            device=device,
            vocal=False
        )
        tds = (tendency_ds.isel(time=1) - tendency_ds.isel(time=0)).sortby("latitude", ascending=False)
        tds.to_netcdf(tendency_path)
        
    # verify tendency reversion
    vds = inference.single_IC_inference(
        model=model,
        n_timesteps=1,
        initial_condition=IC_ds,
        initial_perturbation=None,  # no initial perturbation
        recurrent_perturbation=-tds,  # use tendencies as recurrent perturbation
        device=device,
        vocal=False
    )
    # check if the output is close to the initial condition
    print("Verifying tendency reversion.")
    print("If largest value in following output is close to zero, tendency reversion is working.")
    print("---------------------------")
    print((vds.isel(time=1) - vds.isel(time=0)).max().data_vars)
    print("---------------------------")
else:
    tds = 0
    
### Prepare Recurrent Perturbation ###
    
# rpert will be added to the model output at every timestep
# so we add the negative of the tendency to reverse it
# f will be 0 for all tests but tropical heating
# test below by setting f to 0, model should have approx. static output
rpert = - tds + f
if isinstance(rpert, int) and rpert == 0:
    print("No recurrent perturbation applied.")
    rpert = None

### Run Experiment & Save Output ###
print(f"Running inference for experiment \"{hm24_exp_name}\" with {n_timesteps} timesteps.")
ds = inference.single_IC_inference(
        model=model,
        n_timesteps=n_timesteps,
        initial_condition=IC_ds,
        initial_perturbation=initial_perturbation,
        recurrent_perturbation=rpert,
        device=device,
        vocal=True
    )

# add some metadata
ds = ds.rename({"time": "lead_time"})
ds = ds.assign_coords({"lead_time": lead_times_h})
ds = ds.assign_attrs({"lead_time": "Lead time in hours"})

# save output
print(f"Saving output to {output_path}")
if output_path.exists():
    print(f"Output file already exists. Overwriting.")
    output_path.unlink()
ds.to_netcdf(output_path)



"""
NEW
"""



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
    tendency_file = output_dir / "tendencies" / f"{model_name}_tendency.nc"

    # load the model
    model = general.load_model(model_name)

    # load the initial condition times
    ic_dates = [dt.datetime.strptime(str_date, "%Y/%m/%d %H:%M") for str_date in config["ic_dates"]]

    # interface between model and data
    xr_io = XarrayBackend()

    # get ERA5 data from the ECMWF CDS
    data_source = general.DataSet(
        "<path_to_your_era5_data>"
    )

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