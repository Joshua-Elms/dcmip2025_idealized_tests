from utils import vis, general, model_info
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

print("Visualizing results of HM24 TC Experiment")

config = general.read_config(Path("0.config.yaml"))
    
# unpack config & set paths
IC_params = config["initial_condition_params"]
season = IC_params["season"]
models = config["models"]
pert_params = config["perturbation_params"]
output_dir = Path(config["experiment_dir"]) / config["experiment_name"]
plot_dir = output_dir / "plots"

# convenience vars
n_timesteps = config["n_timesteps"]
amp_vec = pert_params["amp_vec"]
g = 9.81 # m/s^2
cmap_str = pert_params.get("cmap", "viridis")
cmap = plt.get_cmap(cmap_str, len(amp_vec))

for model_name in models:
    print(f"Visualizing {model_name}")
    IC_path = Path(IC_params["HM24_IC_dir"]) / f"{model_name}.nc"
    perturbation_path = Path(pert_params["perturbation_dir"]) / f"{season}_15N_320E_z-regression_{model_name}.nc"
    nc_output_file = output_dir / f"output_{model_name}.nc"
    tendency_file = output_dir / "auxiliary" / f"tendency_{model_name}.nc"
    IC_ds = xr.open_dataset(IC_path)
    ds = xr.open_dataset(nc_output_file)
    tds = xr.open_dataset(tendency_file)
    mean_ds = xr.open_dataset(IC_path)
    mean_ds = general.sort_latitudes(mean_ds, model_name, input=False)
    # some models use only 720 lats instead of 721
    if mean_ds.sizes["lat"] > ds.sizes["lat"]:
        mean_ds = mean_ds.isel(lat=slice(1, ds.sizes["lat"] + 1))
        
    if mean_ds.sizes["time"] > 1:
        mean_ds = mean_ds.isel(time=0) # only need one time slice for mean state
    print("Dividing geopotential Z [m^2/s^2] by 9.8 [m/s^2] to convert to height z [m]")
    for level in model_info.STANDARD_13_LEVELS:
        try:
            levstr = f"z{level}"
            ds[levstr] = ds[levstr] / (g) # convert to geopot. height
            mean_ds[levstr] = mean_ds[levstr] / (g) # convert to geopot. height
        except KeyError:
            continue
        
    print("Dividing surface pressure and mean sea level pressure by 100 to convert from Pa to hPa")
    try:
        ds["sp"] = ds["sp"] / 100.0  # Pa to hPa
        mean_ds["sp"] = mean_ds["sp"] / 100.0  # Pa to hPa
    except KeyError:
        pass
    
    # mean sea level pressure processing
    try:
        ds["msl"] = ds["msl"] / 100.0  # Pa to hPa
        mean_ds["msl"] = mean_ds["msl"] / 100.0  # Pa to hPa
    except KeyError:
        pass

    print(f"Loaded data from {model_name}, beginning visualization.")
    
    #
    ### Begin Tropical Cyclone Visualizations ###
    #
    
    bounding_box = pert_params["bounding_box"] # [lon_min, lon_max, lat_min, lat_max]
    regional_msl = ds["msl"].sel(lon=slice(bounding_box[0], bounding_box[1]), lat=slice(bounding_box[2], bounding_box[3]))
    
    projection = ccrs.Robinson(central_longitude=-90.)

    fig, ax = plt.subplots(figsize=(9,6),subplot_kw={'projection': projection})

    ax.add_feature(cfeature.LAND,edgecolor='0.5',linewidth=0.5,zorder=-1)
    ax.add_feature(cfeature.OCEAN, color='lightblue')
    ax.set_extent([270, 330, 5, 55],crs=ccrs.PlateCarree()) # Atlantic (it=5+)
    ax.coastlines(color='gray')
    ax.set_title(f"{model_name} TC Tracks")
    
    # calculate and plot tracks
    for i, amp in enumerate(amp_vec):
        track_lons = []
        track_lats = []
        for t, time in enumerate(np.arange(0, n_timesteps+1, 6)):
            dat = regional_msl.sel(lead_time=np.timedelta64(time, "h"), amplitude=amp).squeeze().values
            # if amp ==
            # find min mslp location
            min_idx = np.unravel_index(np.argmin(dat, axis=None), dat.shape)
            min_lat = regional_msl.lat.values[min_idx[0]]
            min_lon = regional_msl.lon.values[min_idx[1]]
            track_lats.append(min_lat)
            track_lons.append(min_lon)
        ax.plot(track_lons, track_lats, marker='o', label=f"Amp {amp}", color=cmap(i), transform=ccrs.PlateCarree())

    plt.savefig(plot_dir / f"{model_name}_TC_tracks.png", dpi=300, bbox_inches='tight')
        
    # MSLP anomalies (global)
    titles = [f"{model_name.upper()}: MSLP Anomalies from JAS at t={t*6} hours" for t in range(0, n_timesteps+1)]
    data = ds["msl"].sel(amplitude=10).squeeze() - mean_ds["msl"].squeeze()
    plot_var = f"msl_global_{model_name}"
    vis.create_and_plot_variable_gif(
        data=data,
        plot_var=plot_var,
        iter_var="lead_time",
        iter_vals=np.arange(0, n_timesteps+1),
        plot_dir=plot_dir,
        units="m",
        cmap="bwr",
        titles=titles,
        keep_images=False,
        dpi=300,
        fps=2, 
        vlims=(-50, 50),  # Set vlims for better visualization
        extent=[270, 330, 5, 55],
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
    #
    ### End Tropical Cyclone Visualizations ###
    #
    