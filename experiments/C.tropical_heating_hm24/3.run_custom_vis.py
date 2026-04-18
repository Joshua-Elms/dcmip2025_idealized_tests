"""
These plots are intended to characterize the overall behavior of
the examined models in this experiment, rather than the very narrow
question of "How closely do these models match the magnitude of
Pangu24's response to tropical heating?", as the 2.run_hm24_vis.py
figures effectively answer.
"""

from utils import general, vis, model_info
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import xarray as xr
import numpy as np


config_path = Path(__file__).parent / "0.config.yaml"
config = general.read_config(config_path)
exp_dir = Path(config["experiment_dir"]) / config["experiment_name"]
plot_dir = exp_dir / "plots"
for model in config["models"]:
    model_output_path = exp_dir / f"output_{model}.nc"
    ds = xr.open_dataset(model_output_path).squeeze("init_time")
    if ds["lead_time"].values.dtype == "timedelta64[s]":
        ds["lead_time"] = (ds["lead_time"] / np.timedelta64(1, "h")).astype(int)
    heating_path = exp_dir / "auxiliary" / f"heating_{model}.nc"
    heating_ds = xr.open_dataset(heating_path)
    heating = heating_ds["t500"].isel(time=-1).squeeze().values
    lat, lon = ds["lat"].values, ds["lon"].values
    g = 9.81

    it = 480
    centerlon = 180

    if model == "Pangu24" and (it % 24) != 0:
        print(f"{it} not present in Pangu24, skipping")
        continue

    ds500 = ds.sel(lead_time=it).squeeze()
    z500_mean = ds["z500"].isel(lead_time=0).squeeze().values
    z500_pert = ds500["z500"].values
    u500_pert = ds500["u500"].values
    v500_pert = ds500["v500"].values
    u500_mean = ds["u500"].isel(lead_time=0).squeeze().values
    v500_mean = ds["v500"].isel(lead_time=0).squeeze().values
    pzdat = (z500_pert - z500_mean) / g
    udat = u500_pert - u500_mean
    vdat = v500_pert - v500_mean
    basefield = z500_mean / g

    #     heating = heating_ds["t500"].isel(time=0).squeeze().values

    fig, ax = plt.subplots(
        figsize=(10, 5),
        subplot_kw={"projection": ccrs.Robinson(central_longitude=120.0)},
    )
    ax.coastlines()
    anom_mag = max(pzdat.max(), -pzdat.min())
    im = ax.imshow(
        pzdat,
        cmap="bwr",
        vmin=-anom_mag,
        vmax=anom_mag,
        origin="lower",
        transform=ccrs.PlateCarree(central_longitude=centerlon),
    )
    fig.colorbar(im)
    # plot heating
    cs = ax.contour(
        lon,
        lat,
        heating,
        levels=[0.05],
        colors="r",
        linestyles="dashed",
        linewidths=4,
        transform=ccrs.PlateCarree(),
        alpha=0.75,
    )

    fig.suptitle(f"{model} | Z500 anomalies | {it} hours")

    fig.savefig(plot_dir / f"z500_{model}_{it}_hours.png", dpi=200)

    ### Use this part of the code to start the plot!

    panel_label = ["(A)", "(B)", "(C)"]
    plot_vec = False
    axi = -1
    g = 9.81
    lat, lon = ds["lat"].values, ds["lon"].values
    for it in [120, 240, 480]:
        axi += 1

        # h&m24 plot replication
        # _mean indicates the mean state
        # _pert indicates the perturbed run
        # _anom indicates the anomaly (perturbed run - mean state)
        ds500 = ds.sel(lead_time=it).squeeze()
        z500_mean = ds["z500"].isel(lead_time=0).squeeze().values
        z500_pert = ds500["z500"].values
        u500_pert = ds500["u500"].values
        v500_pert = ds500["v500"].values
        u500_mean = ds["u500"].isel(lead_time=0).squeeze().values
        v500_mean = ds["v500"].isel(lead_time=0).squeeze().values
        pzdat = (z500_pert - z500_mean) / g
        udat = u500_pert - u500_mean
        vdat = v500_pert - v500_mean
        basefield = z500_mean / g

        heating = heating_ds["t500"].isel(time=0).squeeze().values
