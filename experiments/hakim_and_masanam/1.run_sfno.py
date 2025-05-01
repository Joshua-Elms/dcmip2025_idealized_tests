import xarray as xr
import datetime as dt
import torch
import numpy as np
import logging
from earth2mip import networks # type: ignore
from utils import inference
import dotenv
from pathlib import Path
import numpy as np
import yaml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# set up paths
this_dir = Path(__file__).parent

# read configuration
config_path = this_dir / "0.config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    
# load the earth2mip environment variables
dotenv.load_dotenv()

# load the model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}.")
model = networks.get_model("fcnv2_sm").to(device)
print("Model loaded.")


# set data paths
IC_path = config["DJF_mean_sfno_path"]
tendency_path = config["DJF_mean_sfno_tendency_path"]

# read unperturbed initial conditions
IC_ds = xr.open_dataset(IC_path)

# if file exists tendency_path at tendency_path, load it
if tendency_path.exists():
    print(f"Loading cached tendencies from {tendency_path}.")
    ds_tendency = xr.open_dataset(tendency_path)
# else, compute tendencies and save to tendency_path for future use
else:
    print(f"Computing tendencies and saving to {tendency_path}.")
    tendency_ds = inference.single_IC_inference(
        model=model,
        n_timesteps=1,
        initial_condition=IC_ds,
        init_time=None,
        perturbation=None,
        device=device,
        vocal=True
    )
    breakpoint()