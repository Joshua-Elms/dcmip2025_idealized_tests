import xarray as xr
import numpy as np
from earth2mip import networks # type: ignore
from utils import inference_pangu as inference
from pathlib import Path
import numpy as np
import yaml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

print("\n\nRunning Pangu 6-hr model inference for an HM24 experiment.\n\n")

### General Setup ###

# read configuration
this_dir = Path(__file__).parent
config_path = this_dir / "0.config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    
# set up directories
exp_dir = Path(config["experiment_dir"]) / config["hm24_experiment"] / config["experiment_name"] # all data for experiment stored here
tendency_dir = Path(config["experiment_dir"]) / "tendencies" # where to save tendencies
output_path = exp_dir / "pangu_output.nc" # where to save output from inference

# load the model
device : str = config["device"]
assert device in ["cpu", "cuda", "cuda:0"], f"Device must be 'cpu' or 'cuda', got {device}."
print(f"Loading model on {device}.")
model = networks.get_model("pangu_6", device=device)
print("Model loaded.")

# unpack experiment config
n_timesteps : int = config["n_timesteps"]
lead_times_h = np.arange(0, 6*n_timesteps+1, 6)
tendency_reversion : bool = config["tendency_reversion"] 
hm24_exp_name : str = config["hm24_experiment"]
hm24_exp_name_options = ["tropical_heating", "extratropical_cyclone", 
                         "geostrophic_adjustment", "tropical_cyclone"]
assert hm24_exp_name in hm24_exp_name_options, \
    f"Experiment name must be one of {hm24_exp_name_options}, got {hm24_exp_name}."
    
# load initial condition
IC_dir = Path(config["time_mean_IC_dir"])
IC_season = config["IC_season"]
IC_path = IC_dir / f"{IC_season}_ERA5_time_mean_pangu.nc"
IC_ds = xr.open_dataset(IC_path).sortby("latitude", ascending=True)

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
    heating_ds = inference.create_empty_pangu_ds()
    
    # find levels between 1000 and 200 hPa (inclusive)
    levs = IC_ds["level"].values
    levs = levs[(levs <= 1000) & (levs >= 200)]

    # set perturbation temp profile to `heating` field
    heating_ds["T"].loc[dict(level=levs)] = heating

    # set perturbation to zero for all other variables
    zero_vars = ["U", "V", "Z", "Q", "VAR_10U", "VAR_10V", "MSL", "VAR_2T"]
    for var in zero_vars:
        heating_ds[var][:] = 0.
        
    # save heating_ds to file
    print(f"Saving heating dataset to {heating_ds_path}.")
    print(f"Applying tropical heating perturbation: {heating_ds}")
    heating_ds.to_netcdf(heating_ds_path)
    
    # these two get passed to inference.single_IC_inference (via rpert for f)
    initial_perturbation = None
    f = heating_ds
    
elif hm24_exp_name == "extratropical_cyclone":
    etc_pert_path = list(Path(config["perturbation_dir"]).glob(f"cyclone_{IC_season}_*_regression_pangu.nc"))[0]
    etc_pert = xr.open_dataset(etc_pert_path)
    print(f"Loaded extratropical cyclone perturbation from {etc_pert_path}.")
    # pert scaled by user-defined amplitude, separate from HM24 file amp in supplementary/
    amp = config["perturbation_params"]["amp"]
    initial_perturbation = etc_pert * amp # to be added to initial condition before inference
    print(f"Applying ETC perturbation: {initial_perturbation}")
    f = 0 # no recurrent perturbation

elif hm24_exp_name == "geostrophic_adjustment":
    etc_pert_path = list(Path(config["perturbation_dir"]).glob(f"cyclone_{IC_season}_*_regression_pangu.nc"))[0]
    etc_pert = xr.open_dataset(etc_pert_path)
    print(f"Loaded ETC perturbation from {etc_pert_path}.")
    # for geostrophic adjustment test, only z500 is perturbed
    ga_pert = etc_pert * 0
    # pert scaled by user-defined amplitude, separate from HM24 file amp in supplementary/
    amp = config["perturbation_params"]["amp"]
    ga_pert["Z"].loc[{"level": 500}] = etc_pert["Z"].sel(level=500) * amp
    initial_perturbation = ga_pert # to be added to initial condition before inference
    print(f"Applying GA perturbation: {initial_perturbation}")
    f = 0 # no recurrent perturbation

elif hm24_exp_name == "tropical_cyclone":
    tc_pert_path = list(Path(config["perturbation_dir"]).glob(f"hurricane_{IC_season}_*_regression_pangu.nc"))[0]
    tc_pert = xr.open_dataset(tc_pert_path)
    print(f"Loaded tropical cyclone perturbation from {tc_pert_path}.")
    # pert scaled by user-defined amplitude, separate from HM24 file amp in supplementary/
    amp = config["perturbation_params"]["amp"]
    initial_perturbation = tc_pert * amp # to be added to initial condition before inference
    print(f"Applying TC perturbation: {initial_perturbation}")
    f = 0 # no recurrent perturbation
    
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
    tendency_path = tendency_dir / f"{IC_season}_pangu_tendency.nc"
    if tendency_path.exists():
        print(f"Loading cached tendencies from {tendency_path}")
        tds = xr.open_dataset(tendency_path)
    # else, compute tendencies and save to tendency_path for future use
    else:
        print(f"Computing tendencies and saving to {tendency_path}")
        tendency_ds = inference.single_IC_inference(
            model=model,
            n_timesteps=1,
            initial_condition=IC_ds,
            device=device,
            vocal=False
        )
        tds = (tendency_ds.isel(time=1) - tendency_ds.isel(time=0)).sortby("latitude", ascending=True)
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