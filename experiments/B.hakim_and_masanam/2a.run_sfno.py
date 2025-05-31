import xarray as xr
import numpy as np
from earth2mip import networks # type: ignore
from utils import inference
from pathlib import Path
import numpy as np
import yaml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

### General Setup ###

# read configuration
this_dir = Path(__file__).parent
config_path = this_dir / "0.config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    
# set up directories
exp_dir = Path(config["experiment_dir"]) / config["hm24_experiment_params"]["hm24_experiment"] / config["experiment_name"] # all data for experiment stored here
tendency_dir = Path(config["experiment_dir"]) / "tendencies" # where to save tendencies
output_path = exp_dir / "sfno_output.nc" # where to save output from inference

# load the model
device : str = config["device"]
assert device in ["cpu", "cuda", "cuda:0"], f"Device must be 'cpu' or 'cuda', got {device}."
print(f"Loading model on {device}.")
model = networks.get_model("fcnv2_sm").to(device)
print("Model loaded.")

# unpack experiment config
n_timesteps : int = config["n_timesteps"]
lead_times_h = np.arange(0, 6*n_timesteps+1, 6)
exp_params = config["hm24_experiment_params"]
tendency_reversion : bool = exp_params["tendency_reversion"] 
hm24_exp_name : str = exp_params["hm24_experiment"]
hm24_exp_name_options = ["tropical_heating", "extratropical_cyclone", 
                         "geostrophic_adjustment", "tropical_cyclone"]
assert hm24_exp_name in hm24_exp_name_options, \
    f"Experiment name must be one of {hm24_exp_name_options}, got {hm24_exp_name}."
    
# load initial condition
IC_dir = Path(config["time_mean_IC_dir"])
IC_season = config["IC_season"]
IC_path = IC_dir / f"{IC_season}_ERA5_time_mean_sfno.nc"
IC_ds = xr.open_dataset(IC_path).sortby("latitude", ascending=False)

### Setup Specific Experiment ###

# depending on the experiment, grab appropriate parameters & prepare perturbation
if hm24_exp_name == "tropical_heating":
    heating_ds_path = exp_dir / "heating.nc"
    pert_params = exp_params["perturbation_params"]
    km = pert_params["km"]
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
    heating_ds.to_netcdf(heating_ds_path)
    
    # these two get passed to inference.single_IC_inference (via rpert for f)
    initial_perturbation = None
    f = heating_ds
    f = 0
    
elif hm24_exp_name == "extratropical_cyclone":
    pass

elif hm24_exp_name == "geostrophic_adjustment":
    pass

elif hm24_exp_name == "tropical_cyclone":
    pass

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
            vocal=True
        )
        tds = tendency_ds.isel(time=1) - tendency_ds.isel(time=0)
        tds.to_netcdf(tendency_path)
else:
    tds = 0
    
    
### Prepare Recurrent Perturbation ###
    
# rpert will be added to the model output at every timestep
# so we add the negative of the tendency to reverse it
# f will be 0 for all tests but tropical heating
rpert = - tds + f
if rpert == 0:
    rpert = None

# test with below line -- model output should be static
# rpert = -tds

### Run Experiment & Save Output ###

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
print(f"Saving output to {output_path}.")
if output_path.exists():
    print(f"Output file already exists. Overwriting.")
    output_path.unlink()
ds.to_netcdf(output_path)
