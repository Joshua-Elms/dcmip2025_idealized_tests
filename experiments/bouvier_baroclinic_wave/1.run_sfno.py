import xarray as xr
import datetime as dt
import torch
import numpy as np
import logging
from earth2mip import networks # type: ignore
import utils.inference as inference
import experiments.bouvier_baroclinic_wave.initial_conditions.utils as bouvier_utils
import dotenv
from pathlib import Path
import numpy as np
import yaml
from time import perf_counter
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# load the earth2mip environment variables
dotenv.load_dotenv()

# read configuration
this_dir = Path(__file__).parent
config_path = this_dir / "0.config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
ic_params = config["initial_condition_parameters"]

# set up directories
exp_dir = Path(config["experiment_dir"]) / config["experiment_name"] # all data for experiment stored here
exp_dir.mkdir(parents=True, exist_ok=True) # make dir if it doesn't exist
ic_csv_dir = exp_dir / "ic_csv" # contains fort generated ICs, must be processed into nc before used for inference
ic_csv_dir.mkdir(exist_ok=True)
ic_nc_dir = exp_dir / "ic_nc" # contains processed ICs in nc format, ready for inference
ic_nc_dir.mkdir(exist_ok=True)
output_path = exp_dir / "output.nc" # where to save output from inference
log_path = exp_dir / "python.log" # where to save log

# copy config to experiment directory
config_path_exp = exp_dir / "config.yml"
with open(config_path_exp, 'w') as file:
    yaml.dump(config, file)

# set up logging
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s:%(message)s',
    datefmt='%Y-%m-%dH%H:%M:%S'
)
logging.info(f"Experiment: {config['experiment_name']}")
logging.info(f"Config: {config_path}")
print(f"Logging to: {log_path}")
    
# load the model
device = config["inference_parameters"]["device"]
start = perf_counter()
model = networks.get_model("fcnv2_sm").to(device)
end = perf_counter()
logging.info(f"Model loaded in {end-start:.2f} seconds.")

# find the iterable parameter (only one allowed currently)
# see format of ic_params in initial_conditions/bouvier/configs/example.yml
keys, val_pairs = zip(*ic_params.items())
vals, units = zip(*val_pairs)
param_val_pairs = dict(zip(keys, vals))
val_types = [isinstance(v, list) for v in vals]
iter_param_idx = val_types.index(1)
iter_param = keys[iter_param_idx]
iter_vals = np.array(vals[iter_param_idx])
iter_param_units = units[iter_param_idx]
assert sum(val_types) == 1, "Only one iterable parameter allowed"

# generate ic and run inference in same loop
logging.info(f"iterating over {iter_param}: {ic_params[iter_param]}")
ds_list = []
for i, val in enumerate(iter_vals.tolist()): # whichever parameter is iterable
    
    # generate csv initial condition from fort executable
    csv_ic_path = ic_csv_dir / f"ic_{iter_param}={val}.csv"
    csv_kwargs = {
        **param_val_pairs,
        "executable_path": config["fort_executable_path"],
        iter_param: val, # must go after **ic_params to overwrite iter_param value
        "filename": csv_ic_path,
    }
    
    out, err = bouvier_utils.run_fortran_executable(**csv_kwargs)
    logging.info(f"Fortran output: {out}")
    if err:
        logging.info(f"Fortran error: {err}")
    
    # process csv initial condition into netcdf and add some derived variables to it
    nc_ic_path = ic_nc_dir / f"ic_{iter_param}={val}.nc"
    nc_kwargs = {
        **config["processor_parameters"],
        "csv_path": csv_ic_path,
        "nc_path": nc_ic_path,
        "nlat": ic_params["nlat"][0],
        "metadata_dir": Path(config["processor_parameters"]["metadata_dir"]),
        "logf": log_path,
    }
    ic = bouvier_utils.process_individual_fort_file(**nc_kwargs).sortby("lat", ascending=False)
    # ic = xr.open_dataset("/glade/u/home/jmelms/projects/dcmip2025_idealized_tests/experiments/hakim_and_masanam/data/DJF_ERA5_time_mean.nc").sortby("latitude", ascending=False)
    
    # check whether H&M24 tendency reversion is required
    tendency_reversion = config["inference_parameters"]["tendency_reversion"]
    
    # if so, calculate the tendency for this IC
    if tendency_reversion:
        tendency_ds = inference.single_IC_inference(
            model=model,
            n_timesteps=1,
            initial_condition=ic,
            device=device,
            vocal=True
        )
        tds = (tendency_ds.isel(time=1) - tendency_ds.isel(time=0)).sortby("latitude", ascending=False)
        rpert = -tds  # recurrent perturbation is the negative of the tendency
        
        # Run a test to verify the recurrent perturbation mechanism
        verification_ds = inference.single_IC_inference(
            model=model,
            n_timesteps=1,
            initial_condition=ic,
            recurrent_perturbation=rpert,
            device=device,
            vocal=True
        )
        # We should find that: verification_ds.isel(time=1) ≈ verification_ds.isel(time=0)
        max_e = (verification_ds.isel(time=1) - verification_ds.isel(time=0)).max()
        print(f"RMSE between t=1 and t=0 with perturbation: {max_e}")
    # else, set the recurrent perturbation to None
    else:
        rpert = None
        
    # optionally perturb the initial condition
    pert_params = config["perturbation_parameters"]
    if pert_params["enabled"]:
        ylat = pert_params["center_latitude"]
        xlon = pert_params["center_longitude"]
        locRad = pert_params["localization_radius"]
        u_pert_base = pert_params["u_wind_pert_base"]
        
        # set up the perturbation
        upert = inference.gen_baroclinic_wave_perturbation(
            ic.lat, ic.lon, ylat, xlon, u_pert_base, locRad
        )
        initial_perturbation = inference.create_empty_sfno_ds()

        # set perturbation u-wind profile to `upert` field
        initial_perturbation["U"][:] = upert
        initial_perturbation["VAR_10U"][:] = upert
        initial_perturbation["VAR_100U"][:] = upert
        
        # set perturbation to zero for all other variables
        zero_vars = ["T", "V", "Z", "R", "VAR_10V", "VAR_100V", "SP", "MSL", "TCW", "VAR_2T"]
        for var in zero_vars:
            initial_perturbation[var][:] = 0.
            
        initial_perturbation.to_netcdf(
            ic_nc_dir / f"initial_perturbation.nc"
        )
        print(f"Perturbation saved to {ic_nc_dir / 'initial_perturbation.nc'}")
            
    else: 
        initial_perturbation = None
        
    single_ds = inference.single_IC_inference(
        model=model,
        n_timesteps=config["inference_parameters"]["n_steps"],
        initial_condition=ic,
        initial_perturbation=initial_perturbation,
        recurrent_perturbation=rpert,
        device=device,
        vocal=True, 
        )
    
    ds_list.append(single_ds)
    
# save output
ds_out = xr.concat(ds_list, dim=iter_param)
ds_out = ds_out.rename({"time": "lead_time"})
ds_out = ds_out.assign_coords({iter_param: iter_vals})
ds_out = ds_out.assign_attrs({f"{iter_param} units": iter_param_units})       
ds_out.to_netcdf(output_path)

logging.info(f"ds_out shape: {ds_out.dims}")
logging.info(f"Saved ds of size {ds_out.nbytes/1e9:.2f} GB to {output_path}")
logging.info("Finished.")
print("Finished.")