from utils import vis, model_info
from utils import general
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

print("Visualizing results of HM24 GA Experiment")

config = general.read_config(Path("0.config.yaml"))

# unpack config & set paths
season = config["initial_condition_params"]["season"]
models = config["models"]
output_dir = Path(config["experiment_dir"]) / config["experiment_name"]
plot_dir = output_dir / "plots"

# convenience vars
n_timesteps = config["n_timesteps"]
lead_times_h = np.arange(0, 6 * n_timesteps + 1, 6)
g = 9.81  # m/s^2

for model_name in models:
    if model_name == "Pangu24":
        exit()
    print(f"Visualizing {model_name}")
    IC_path = (
        Path(config["initial_condition_params"]["HM24_IC_dir"]) / f"{model_name}.nc"
    )
    perturbation_path = (
        Path(config["perturbation_params"]["perturbation_dir"])
        / f"{season}_40N_150E_z-regression_{model_name}.nc"
    )
    nc_output_file = output_dir / f"output_{model_name}.nc"
    ds = xr.open_dataset(nc_output_file).sortby("lat", ascending=False)
    mean_ds = xr.open_dataset(IC_path).sortby("lat", ascending=False)
    print("Dividing geopotential Z [m^2/s^2] by 9.8 [m/s^2] to convert to height z [m]")
    for level in model_info.STANDARD_13_LEVELS:
        levstr = f"z{level}"
        ds[levstr] = ds[levstr] / (g)  # convert to geopot. height
        mean_ds[levstr] = mean_ds[levstr] / (g)  # convert to geopot. height

    print(f"Loaded data from {model_name}, beginning visualization.")

    #
    ### Begin Extratropical Cyclone Visualizations ###
    #

    ### begin HM24 Fig. 3
    font = {"family": "DejaVu Sans", "weight": "bold", "size": 12}
    matplotlib.rc("font", **font)

    plot_vec = False
    plot_vec = True

    projection = ccrs.Robinson(central_longitude=-90.0)
    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11 * 2, 8.5 * (1)),
        subplot_kw={"projection": projection},
        layout="constrained",
    )

    panel_label = ["(A)", "(D)"]
    figfile = plot_dir / "geo_adjust_500_40N.pdf"

    axi = -1
    lat, lon = ds["lat"].values, ds["lon"].values
    for it in [0, 6]:
        axi += 1

        z500_mean = mean_ds["z500"].squeeze().values
        u500_mean = mean_ds["u500"].squeeze().values
        v500_mean = mean_ds["v500"].squeeze().values
        z500_pert = ds["z500"].sel(lead_time=np.timedelta64(it, "h")).squeeze().values
        u500_pert = ds["u500"].sel(lead_time=np.timedelta64(it, "h")).squeeze().values
        v500_pert = ds["v500"].sel(lead_time=np.timedelta64(it, "h")).squeeze().values
        pzdat = z500_pert - z500_mean
        udat = u500_pert - u500_mean
        vdat = v500_pert - v500_mean
        basefield = z500_pert
        
        breakpoint()

        dcint = 20
        ncint = 5
        vscale = 150  # vector scaling (counterintuitive:smaller=larger arrows)

        if plot_vec:
            # Plot vectors on the map
            latskip = 7
            lonskip = 7
            alpha = 1.0
            col = "g"
            cs = ax[axi].quiver(
                lon[::lonskip],
                lat[::latskip],
                udat[::latskip, ::lonskip],
                vdat[::latskip, ::lonskip],
                transform=ccrs.PlateCarree(),
                scale=vscale,
                color=col,
                alpha=alpha,
            )
            if "_40N" in str(perturbation_path):
                qk = ax[axi].quiverkey(
                    cs,
                    0.65,
                    0.01,
                    10.0,
                    r"$10~ m/s$",
                    labelpos="E",
                    coordinates="figure",
                    color=col,
                )
            elif "_0N" in str(perturbation_path):
                qk = ax[axi].quiverkey(
                    cs,
                    0.5,
                    0.01,
                    10.0,
                    r"$10~ m/s$",
                    labelpos="E",
                    coordinates="figure",
                    color=col,
                )

        # mean state or full field
        alpha = 1.0
        cints = np.arange(4800, 6000, 60.0)
        cs = ax[axi].contour(
            lon,
            lat,
            basefield,
            levels=cints,
            colors="0.5",
            linewidths=2.0,
            transform=ccrs.PlateCarree(),
            alpha=alpha,
            zorder=0,
        )
        # perturbations
        alpha = 0.75
        cints = list(np.arange(-ncint * dcint, -dcint + 0.001, dcint)) + list(
            np.arange(dcint, ncint * dcint + 0.001, dcint)
        )
        cints_neg = list(np.arange(-ncint * dcint, -dcint + 0.001, dcint))
        cints_pos = list(np.arange(dcint, ncint * dcint + 0.001, dcint))
        cs = ax[axi].contour(
            lon,
            lat,
            pzdat,
            levels=cints_neg,
            colors="b",
            linestyles="solid",
            transform=ccrs.PlateCarree(),
            alpha=alpha,
        )
        cs = ax[axi].contour(
            lon,
            lat,
            pzdat,
            levels=cints_pos,
            colors="r",
            linestyles="solid",
            transform=ccrs.PlateCarree(),
            alpha=alpha,
        )

        # colorize land
        ax[axi].add_feature(cfeature.LAND, edgecolor="0.5", linewidth=0.5, zorder=-1)

        ax[axi].coastlines(color="gray")
        gl = ax[axi].gridlines(
            crs=ccrs.PlateCarree(),
            linewidth=1,
            color="gray",
            alpha=0.5,
            linestyle="--",
            draw_labels=True,
        )
        if axi == 1:
            gl.bottom_labels = True
        else:
            gl.bottom_labels = False

        ax[axi].set_extent([120, 200, 25, 55], crs=ccrs.PlateCarree())  # Pacific
        ax[axi].text(113, 25, panel_label[axi], transform=ccrs.PlateCarree())

        print("zmin for t=", it, ":", np.min(pzdat), "m")

        fig.tight_layout()
        plt.savefig(figfile, dpi=300, bbox_inches="tight")
        ### end HM24 Fig. 3

        #
        ### End Extratropical Cyclone Visualizations ###
        #
