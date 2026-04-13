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

    it = 30
    
    if model == "Pangu24":
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
        transform=ccrs.PlateCarree(central_longitude=180),
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

    fig.savefig(plot_dir / f"z500_{model}_{it}_hours.png", dpi=200)

    # # z1000
    # nt = config["n_timesteps"]
    # titles = [f"{model}: $Z_{{1000}}$ at t={t*6} hours" for t in range(0, nt + 1)]
    # data = ds["z1000"] / (9.8 * 10)
    # plot_var = f"z1000_{model}"
    # vis.create_and_plot_variable_gif(
    #     data=data,
    #     plot_var=plot_var,
    #     iter_var="lead_time",
    #     iter_vals=np.arange(0, nt + 1),
    #     plot_dir=plot_dir,
    #     units="dam",
    #     cmap="PRGn",
    #     titles=titles,
    #     keep_images=False,
    #     dpi=300,
    #     fps=1,
    #     vlims=(-50, 50),  # Set vlims for better visualization
    #     central_longitude=180.0,
    # )
    # print(f"Made {plot_var}.gif.")

    # # z500_anom
    # nt = config["n_timesteps"]
    # plot_var = f"z500_anom_{model}"
    # titles = [
    #     f"{plot_var} at t={t*model_info.MODEL_TIME_STEP_HOURS[model]} hours"
    #     for t in range(15, nt + 1)
    # ]
    # data = np.abs(ds["z500"] - ds["z500"].isel(lead_time=0))[15:]
    # vis.create_and_plot_variable_gif(
    #     data=data,
    #     plot_var=plot_var,
    #     iter_var="lead_time",
    #     iter_vals=np.arange(6),
    #     plot_dir=plot_dir,
    #     units="J/kg",
    #     cmap="Reds",
    #     titles=titles,
    #     keep_images=False,
    #     dpi=300,
    #     fps=1 / 4,
    #     vlims=(0, 60),  # Set vlims for better visualization
    #     central_longitude=180.0,
    # )
    # print(f"Made {plot_var}.gif.")

    # # heating w/ cartopy borders
    # fig, ax = plt.subplots(
    #     figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()}
    # )
    # heating_ds["t1000"].isel(time=0).plot(
    #     ax=ax, cmap="RdBu", cbar_kwargs={"label": "Heating (K/day)"}
    # )
    # ax.coastlines()
    # ax.set_title(f"{model}: Heating")
    # plt.savefig(plot_dir / f"heating_{model}.png", dpi=200)
    # plt.close(fig)
    # print(f"Made heating_{model}.png")

    # plot from paper
    ### begin HM24 fig. 1

    # projection = ccrs.Robinson(central_longitude=120.0)
    # fig, ax = plt.subplots(
    #     nrows=3,
    #     ncols=1,
    #     figsize=(11 * 2, 8.5 * 2),
    #     subplot_kw={"projection": projection},
    #     layout="constrained",
    # )

    # panel_label = ["(A)", "(B)", "(C)"]
    # plot_vec = False
    # axi = -1
    # g = 9.81
    # lat, lon = ds["lat"].values, ds["lon"].values
    # for it in [120, 240, 480]:
    #     axi += 1

    #     # h&m24 plot replication
    #     # _mean indicates the mean state
    #     # _pert indicates the perturbed run
    #     # _anom indicates the anomaly (perturbed run - mean state)
    #     ds500 = ds.sel(lead_time=it).squeeze()
    #     z500_mean = ds["z500"].isel(lead_time=0).squeeze().values
    #     z500_pert = ds500["z500"].values
    #     u500_pert = ds500["u500"].values
    #     v500_pert = ds500["v500"].values
    #     u500_mean = ds["u500"].isel(lead_time=0).squeeze().values
    #     v500_mean = ds["v500"].isel(lead_time=0).squeeze().values
    #     pzdat = (z500_pert - z500_mean) / g
    #     udat = u500_pert - u500_mean
    #     vdat = v500_pert - v500_mean
    #     basefield = z500_mean / g

    #     heating = heating_ds["t500"].isel(time=0).squeeze().values
