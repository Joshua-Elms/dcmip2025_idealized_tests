"""
This plot combines the approaches of HM24 (2.run_hm24_vis.py) and
3.run_custom_vis.py by using contours as in the former, but allowing
them to be chosen uniquely for each model and lead time, as in the latter.

This should make it easier to answer one of key questions about these
models in the Discussion: do they produce a great-circle wavetrain in
response to the steady heating applied? The imshow-based visualizations
are much smoother and difficult to characterize.
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

N_CONTOURS = 5  # number of positive (and negative) contour lines per panel

LEAD_TIMES = [120, 240, 480]

for model in config["models"]:
    model_output_path = exp_dir / f"output_{model}.nc"
    ds = xr.open_dataset(model_output_path).squeeze("init_time")
    if ds["lead_time"].values.dtype == "timedelta64[s]":
        ds["lead_time"] = (ds["lead_time"] / np.timedelta64(1, "h")).astype(int)
    heating_path = exp_dir / "auxiliary" / f"heating_{model}.nc"
    heating_ds = xr.open_dataset(heating_path)

    projection = ccrs.Robinson(central_longitude=120.0)
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(11 * 2, 8.5 * 2),
        subplot_kw={"projection": projection},
        layout="constrained",
    )

    g = 9.81
    lat, lon = ds["lat"].values, ds["lon"].values
    heating = (
        heating_ds["t500"].isel(time=-1).squeeze().values
    )  # needs to be time=-1 because 2-step models have heating=0 initially
    z500_mean = ds["z500"].isel(lead_time=0).squeeze().values
    basefield = z500_mean / g

    for axi, it in enumerate(LEAD_TIMES):
        ax = axes[axi]

        if model == "Pangu24" and (it % 24) != 0:
            print(
                f"{model}: t={it} h not present (not multiple of 24), skipping panel."
            )
            ax.set_visible(False)
            continue

        ds500 = ds.sel(lead_time=it).squeeze()
        z500_pert = ds500["z500"].values
        pzdat = (z500_pert - z500_mean) / g

        # --- Adaptive symmetric contour intervals ---
        anom_mag = max(abs(pzdat.max()), abs(pzdat.min()))
        dcint = anom_mag / N_CONTOURS
        cints_neg = np.arange(-N_CONTOURS * dcint, 0, dcint)
        cints_pos = np.arange(dcint, (N_CONTOURS + 0.5) * dcint, dcint)

        # Mean-state Z500 contours (grey)
        cints_base = np.arange(4800, 6000, 60.0)
        cs_base = ax.contour(
            lon,
            lat,
            basefield,
            levels=cints_base,
            colors="0.5",
            linewidths=0.8,
            transform=ccrs.PlateCarree(),
            alpha=1.0,
        )
        ax.clabel(cs_base, fontsize=6, inline=True, fmt="%g")

        # Negative anomaly contours (blue, solid)
        cs_neg = ax.contour(
            lon,
            lat,
            pzdat,
            levels=cints_neg,
            colors="b",
            linestyles="solid",
            linewidths=2.0,
            transform=ccrs.PlateCarree(),
        )
        ax.clabel(cs_neg, fontsize=9, inline=True, fmt="%.2g", inline_spacing=5)

        # Positive anomaly contours (red, solid)
        cs_pos = ax.contour(
            lon,
            lat,
            pzdat,
            levels=cints_pos,
            colors="r",
            linestyles="solid",
            linewidths=2.0,
            transform=ccrs.PlateCarree(),
        )
        ax.clabel(cs_pos, fontsize=9, inline=True, fmt="%.2g", inline_spacing=5)

        # Heating outline (dashed red)
        ax.contour(
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

        # Land shading
        ax.add_feature(cfeature.LAND, edgecolor="0.5", linewidth=0.5, zorder=-1)

        # Gridlines
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            linewidth=1.0,
            color="gray",
            alpha=0.5,
            linestyle="--",
            draw_labels=True,
        )
        gl.top_labels = False
        if axi != len(LEAD_TIMES) - 1:
            gl.bottom_labels = False
        gl.xlabels_left = True

        # Panel label — large and bold, showing lead time
        ax.text(
            0.01,
            0.04,
            f"t = {it}h",
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )

        print(f"{model} | t={it} h | anom_mag={anom_mag:.3f} m | dcint={dcint:.3f} m")

    fig.suptitle(f"Tropical Heating: {model}", fontsize=22, fontweight="bold")
    fname = f"heating_500z_{model}_adaptive.png"
    plt.savefig(plot_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname} to {plot_dir}.")
