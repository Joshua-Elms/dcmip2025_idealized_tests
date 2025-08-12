from pathlib import Path
from utils_E2S import general
import xarray as xr
import matplotlib.pyplot as plt
from utils import vis
import numpy as np
import cartopy.crs as ccrs


config_path = Path(__file__).parent / "0.config.yaml"
config = general.read_config(config_path)
exp_dir = Path(config["experiment_dir"]) / config["experiment_name"]
plot_dir = exp_dir / "plots"
for model in config["models"]:
    model_output_path = exp_dir / f"output_{model}.nc"
    ds = xr.open_dataset(model_output_path).squeeze("init_time")
    heating_path = exp_dir / "auxiliary" / f"heating_{model}.nc"
    heating_ds = xr.open_dataset(heating_path)

    # Z850
    nt = config["n_timesteps"]
    titles = [f"{model}: $Z_{{850}}$ at t={t*6} hours" for t in range(0, nt+1)]
    data = ds["z850"] / (9.8*10)
    plot_var = f"Z850_{model}"
    vis.create_and_plot_variable_gif(
        data=data,
        plot_var=plot_var,
        iter_var="lead_time",
        iter_vals=np.arange(0, nt+1),
        plot_dir=plot_dir,
        units="dam",
        cmap="PRGn",
        titles=titles,
        keep_images=False,
        dpi=300,
        fps=2, 
        vlims=(450, 600),  # Set vlims for better visualization
        central_longitude=180.0,
    )
    print(f"Made {plot_var}.gif.")
    
    # Z850_anom
    nt = config["n_timesteps"]
    plot_var = f"Z850_anom_{model}"
    titles = [f"{model}: {plot_var} at t={t*6} hours" for t in range(0, nt+1)]
    data = (ds["z850"] - ds["z850"].isel(lead_time=0)) / (9.8*10)
    vis.create_and_plot_variable_gif(
        data=data,
        plot_var=plot_var,
        iter_var="lead_time",
        iter_vals=np.arange(0, nt+1),
        plot_dir=plot_dir,
        units="dam",
        cmap="PRGn",
        titles=titles,
        keep_images=False,
        dpi=300,
        fps=2, 
        vlims=(-5, 5),  # Set vlims for better visualization
        central_longitude=180.0,
    )
    print(f"Made {plot_var}.gif.")
    
    # heating w/ cartopy borders
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()})
    heating_ds["t1000"].isel(time=0).plot(ax=ax, cmap="RdBu", cbar_kwargs={"label": "Heating (K/day)"})
    ax.coastlines()
    ax.set_title(f"{model}: Heating")
    plt.savefig(plot_dir / f"heating_{model}.png", dpi=200)
    plt.close(fig)
    print(f"Made heating_{model}.png")