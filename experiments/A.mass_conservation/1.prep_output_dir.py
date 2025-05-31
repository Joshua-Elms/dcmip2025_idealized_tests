""" This script prepares the output directory for an experiment. """
from pathlib import Path
import yaml

# read configuration
this_dir = Path(__file__).parent
config_path = this_dir / "0.config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# set up directories
exp_dir = Path(config["experiment_dir"]) / config["experiment_name"] # all data for experiment stored here

# check if YOURUSERNAME appears in exp_dir; error-out if it does
if "YOURUSERNAME" in str(exp_dir):
    raise ValueError("Please replace 'YOURUSERNAME' in 0.config.yaml with your actual username.")

exp_dir.mkdir(parents=True, exist_ok=True) # make dir if it doesn't exist
plot_dir = exp_dir / "plots" # save figures here
plot_dir.mkdir(parents=True, exist_ok=True) # make dir if it doesn't exist

# copy config to experiment directory
config_path_exp = exp_dir / "config.yaml"
with open(config_path_exp, 'w') as file:
    yaml.dump(config, file)