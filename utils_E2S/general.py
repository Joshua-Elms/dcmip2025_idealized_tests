import xarray as xr
from datetime import datetime
import numpy as np
from earth2studio.utils.type import TimeArray, VariableArray
from earth2studio.models.px.base import PrognosticModel
import earth2studio.models.px
from pathlib import Path
import yaml
import shutil
from dotenv import load_dotenv
from time import perf_counter
import torch

class DataSet:
    """An in-memory xarray dataset data source.

    Parameters
    ----------
    dataset : xr.Dataset
        The xarray dataset to use as data source.
    """

    def __init__(self, dataset: xr.Dataset):
        self.ds = dataset

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for.
        variable : str | list[str] | VariableArray
            Strings or list of strings that refer to variables to return.

        Returns
        -------
        xr.DataArray
            Loaded data array
        """
        # loop over variables and concatenate the data arrays
        da_list = [self.ds[v].sel(time=np.atleast_1d(time)) for v in variable]
        da = xr.concat(da_list, dim="variable")
        da = da.assign_coords(variable=variable)
        # reorder to time variable lat lon
        da = da.transpose("time", "variable", "lat", "lon")
        return da
    
def read_config(config_path: Path) -> dict: 
    """Read the configuration file."""

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    # Check for required keys
    required_keys = ["experiment_dir", "experiment_name", "device", "n_timesteps", "models"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required key in config: {key}\nPlease set it and try again.")
    
    # add config path to config object for reference in downstream functions
    if "config_path" not in config:
        config["config_path"] = config_path
        
    return config

def prepare_output_directory(config: dict) -> Path:
    """Prepare the output directory for an experiment."""
    # set up directories
    exp_dir = Path(config["experiment_dir"]) / config["experiment_name"] # all data for experiment stored here
    
    if exp_dir.exists():
        raise FileExistsError(f"Experiment directory '{exp_dir}' already exists. Please delete it or change experiment_name.")

    exp_dir.mkdir(parents=True, exist_ok=False) # make dir if it doesn't exist
    if "experiment_subdirectories" in config:
        for subdir in config["experiment_subdirectories"]:
            (exp_dir / subdir).mkdir()  # create subdirectories if specified in config, e.g. "plots", "tendencies"

    # copy config to experiment directory
    config_path_exp = exp_dir / "config.yaml"
    shutil.copy(config["config_path"], config_path_exp)
        
    # let user know where to find config
    print(f"Ready for experiment output at '{exp_dir}'.")
    
    return exp_dir

def load_model(model_name: str) -> PrognosticModel:
    """Load a model by name. Currently loads default model weights from cache, or downloads them to cache if not present."""
    load_dotenv()
    models = {"SFNO", "Pangu6", "GraphCastOperational", "FuXi"}
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' is not supported. Supported models are: {models}.")
    model_class = getattr(earth2studio.models.px, model_name)
    start = perf_counter()
    print(f"Loading model '{model_name}'...")
    package = model_class.load_default_package()
    model = model_class.load_model(package)
    end = perf_counter()
    print(f"Model '{model_name}' loaded in {end - start:.2f} seconds.")
    return model

def latitude_weighted_mean(da, latitudes, device="cpu"):
    """
    Calculate the latitude weighted mean of a variable using torch operations on GPU.
    Needs tests to ensure it works correctly.
    
    Parameters:
    -----------
    da : xarray.DataArray or torch.Tensor
        The data to average
    latitudes : xarray.DataArray or numpy.ndarray
        The latitude values
        
    Returns:
    --------
    torch.Tensor
        The latitude-weighted mean
    """
    # Convert inputs to torch tensors if needed
    coords = {dim: da[dim] for dim in da.dims if dim not in ['latitude', 'longitude', 'lat', 'lon']}
    if isinstance(da, xr.DataArray):
        da = torch.from_numpy(da.values)
    if isinstance(latitudes, xr.DataArray):
        latitudes = latitudes.values
    
    # Move to GPU if available
    da = da.to(device)
    
    # Calculate weights
    lat_radians = torch.from_numpy(np.deg2rad(latitudes)).to(device)
    weights = torch.cos(lat_radians) / (torch.sum(torch.cos(lat_radians)) * da.shape[-1])
    
    # Expand weights to match data dimensions
    weights = weights.view(1, -1, 1)  # Add dims for broadcasting
    
    # Calculate weighted mean
    weighted_data = da * weights
    averaged = weighted_data.nansum(dim=(-2, -1))  # Average over lat, lon dimensions
    return xr.DataArray(averaged.cpu().numpy(), coords=coords)

if __name__ == "__main__":
    print(load_model("SFNO"))  # Example usage