from utils import vis, general, model_info
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

print("Checking tendency reversion")

config = general.read_config(Path("0.config.yaml"))
    
# unpack config & set paths
IC_params = config["initial_condition_params"]
models = config["models"]
output_dir = Path(config["experiment_dir"]) / config["experiment_name"]
plot_dir = output_dir / "plots"

# convenience vars
n_timesteps = config["n_timesteps"]
g = 9.81 # m/s^2

for model_name in models:
    print(f"Visualizing {model_name}")
    nc_output_file = output_dir / f"{model_name}_output.nc"
    tendency_file = output_dir / "auxiliary" / f"{model_name}_tendency.nc"
    ds = xr.open_dataset(nc_output_file)
    tds = xr.open_dataset(tendency_file)

    print("Dividing geopotential Z [m^2/s^2] by 9.8 [m/s^2] to convert to height z [m]")
    for level in model_info.STANDARD_13_LEVELS:
        try:
            levstr = f"z{level}"
            ds[levstr] = ds[levstr] / (g) # convert to geopot. height
            tds[levstr] = tds[levstr] / (g) # convert to geopot. height
        except KeyError:
            continue
        
    print("Dividing surface pressure and mean sea level pressure by 100 to convert from Pa to hPa")
    try:
        ds["sp"] = ds["sp"] / 100.0  # Pa to hPa
        tds["sp"] = tds["sp"] / 100.0  # Pa to hPa
    except KeyError:
        pass
    
    # mean sea level pressure processing
    try:
        ds["msl"] = ds["msl"] / 100.0  # Pa to hPa
        tds["msl"] = tds["msl"] / 100.0  # Pa to hPa
    except KeyError:
        pass

    print(f"Loaded data from {model_name}, beginning visualization.")
    
    ### make gifs
    
    # Z500 anomalies (global)
    if "z500_anom_global" in config["analysis_params"]["plots"]:
        titles = [f"{model_name.upper()}: $Z_{{500}}$ Anomalies at t={t*6} hours" for t in range(0, n_timesteps+1)]
        data = ds["z500"].squeeze() - (ds["z500"].isel(lead_time=0)).squeeze()
        plot_var = f"z500_anom_global_{model_name}"
        vis.create_and_plot_variable_gif(
            data=data,
            plot_var=plot_var,
            iter_var="lead_time",
            iter_vals=np.arange(0, n_timesteps+1),
            plot_dir=plot_dir,
            units="m",
            cmap="PuOr",
            titles=titles,
            keep_images=False,
            dpi=300,
            fps=2, 
            vlims=(-10, 10),  # Set vlims for better visualization
            central_longitude=180.0,
            fig_size = (7.5, 3.5),
            adjust = {
                "top": 0.97,
                "bottom": 0.01,
                "left": 0.09,
                "right": 0.87,
                "hspace": 0.0,
                "wspace": 0.0,
            },
            cbar_kwargs = {
                "rotation": "horizontal",
                "y": -0.02,
                "horizontalalignment": "right",
                "labelpad": -34.5,
                "fontsize": 9
            },
        )

        print(f"Made {plot_var}.gif.")
    
    # T2M anomalies (global)
    if "t2m_anom_global" in config["analysis_params"]["plots"]:
        titles = [f"{model_name.upper()}: 2-Meter Temperature Anomalies at t={t*6} hours" for t in range(0, n_timesteps+1)]
        data = ds["t2m"].squeeze() - (ds["t2m"].isel(lead_time=0)).squeeze()
        plot_var = f"t2m_anom_global_{model_name}"
        vis.create_and_plot_variable_gif(
            data=data,
            plot_var=plot_var,
            iter_var="lead_time",
            iter_vals=np.arange(0, n_timesteps+1),
            plot_dir=plot_dir,
            units="m",
            cmap="PRGn",
            titles=titles,
            keep_images=False,
            dpi=300,
            fps=2, 
            vlims=(-3, 3),  # Set vlims for better visualization
            central_longitude=180.0,
            fig_size = (7.5, 3.5),
            adjust = {
                "top": 0.97,
                "bottom": 0.01,
                "left": 0.09,
                "right": 0.87,
                "hspace": 0.0,
                "wspace": 0.0,
            },
            cbar_kwargs = {
                "rotation": "horizontal",
                "y": -0.02,
                "horizontalalignment": "right",
                "labelpad": -34.5,
                "fontsize": 9
            },
        )

        print(f"Made {plot_var}.gif.")
    
    # MSLP anomalies (global)
    if "mslp_anom_global" in config["analysis_params"]["plots"]:
        titles = [f"{model_name.upper()}: MSLP Anomalies at t={t*6} hours" for t in range(0, n_timesteps+1)]
        data = ds["msl"].squeeze() - (ds["msl"].isel(lead_time=0)).squeeze()
        plot_var = f"msl_global_{model_name}"
        vis.create_and_plot_variable_gif(
            data=data,
            plot_var=plot_var,
            iter_var="lead_time",
            iter_vals=np.arange(0, n_timesteps+1),
            plot_dir=plot_dir,
            units="m",
            cmap="seismic",
            titles=titles,
            keep_images=False,
            dpi=300,
            fps=2, 
            vlims=(-5, 5),  # Set vlims for better visualization
            central_longitude=180.0,
            fig_size = (7.5, 3.5),
            adjust = {
                "top": 0.97,
                "bottom": 0.01,
                "left": 0.09,
                "right": 0.87,
                "hspace": 0.0,
                "wspace": 0.0,
            },
            cbar_kwargs = {
                "rotation": "horizontal",
                "y": -0.02,
                "horizontalalignment": "right",
                "labelpad": -34.5,
                "fontsize": 9
            },
        )

        print(f"Made {plot_var}.gif.")
    