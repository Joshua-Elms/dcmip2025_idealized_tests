from pathlib import Path
import shutil
import yaml

# read configuration
this_dir = Path(__file__).parent
config_path = this_dir / "0.config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
ic_params = config["initial_condition_parameters"]

# set up directories
exp_dir = Path(config["experiment_dir"]) / config["experiment_name"] # all data for experiment stored here

# check if YOURUSERNAME appears in exp_dir; error-out if it does
if "YOURUSERNAME" in str(exp_dir):
    raise ValueError("Please replace 'YOURUSERNAME' in 0.config.yaml with your actual username.")

exp_dir.mkdir(parents=True, exist_ok=True) # make dir if it doesn't exist
plot_dir = exp_dir / "plots" # save figures here
plot_dir.mkdir(parents=True, exist_ok=True) # make dir if it doesn't exist
ic_csv_dir = exp_dir / "ic_csv" # contains fort generated ICs, must be processed into nc before used for inference
ic_csv_dir.mkdir(exist_ok=True)
ic_nc_dir = exp_dir / "ic_nc" # contains processed ICs in nc format, ready for inference
ic_nc_dir.mkdir(exist_ok=True)
output_path = exp_dir / "output.nc" # where to save output from inference

# copy config to experiment directory
config_path_exp = exp_dir / "config.yaml"
shutil.copy(config_path, config_path_exp)