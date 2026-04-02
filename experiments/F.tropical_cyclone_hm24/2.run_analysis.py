from utils import vis, general, model_info
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def tc_vitals(tc_sfc, lat, lon):
    """Function taken directly from https://github.com/modons/DL-weather-dynamics/blob/main/plot_paper_hurricane.py#L59"""
    latmin = []
    lonmin = []
    pmin = []
    for it in range(tc_sfc.shape[0]):
        minp = np.where(tc_sfc[it, :, :] == np.min(tc_sfc[it, :, :]))
        latmin.append(lat[minp[0][0]])
        lonmin.append(lon[minp[1][0]])
        pmin.append(np.min(tc_sfc[it, :, :]))
    return np.array(latmin), np.array(lonmin), np.array(pmin)


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
g = 9.81  # m/s^2
cmap_str = pert_params.get("cmap", "viridis")
cmap = plt.get_cmap(cmap_str, len(amp_vec))
cols = []
for key in mcolors.BASE_COLORS:
    cols.append(key)

for model_name in models:
    print(f"Visualizing {model_name}")
    IC_path = Path(IC_params["HM24_IC_dir"]) / f"{model_name}.nc"
    perturbation_path = (
        Path(pert_params["perturbation_dir"])
        / f"{season}_15N_320E_z-regression_{model_name}.nc"
    )
    nc_output_file = output_dir / f"output_{model_name}.nc"
    tendency_file = (
        output_dir / "auxiliary" / f"tendency_{model_name}_amp={amp_vec[-1]}.nc"
    )
    ds = xr.open_dataset(nc_output_file)
    mean_ds = xr.open_dataset(IC_path)
    mean_ds = general.sort_latitudes(mean_ds, model_name, input=False)
    # some models use only 720 lats instead of 721
    if mean_ds.sizes["lat"] > ds.sizes["lat"]:
        mean_ds = mean_ds.isel(lat=slice(1, ds.sizes["lat"] + 1))

    if mean_ds.sizes["time"] > 1:
        mean_ds = mean_ds.isel(time=0)  # only need one time slice for mean state
    print("Dividing geopotential Z [m^2/s^2] by 9.8 [m/s^2] to convert to height z [m]")
    for level in model_info.STANDARD_13_LEVELS:
        try:
            levstr = f"z{level}"
            ds[levstr] = ds[levstr] / (g)  # convert to geopot. height
            mean_ds[levstr] = mean_ds[levstr] / (g)  # convert to geopot. height
        except KeyError:
            continue

    print(
        "Dividing surface pressure and mean sea level pressure by 100 to convert from Pa to hPa"
    )
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

    bounding_box = pert_params["bounding_box"]  # [lon_min, lon_max, lat_min, lat_max]
    regional_msl = ds["msl"].sel(
        lon=slice(bounding_box[0], bounding_box[1]),
        lat=slice(bounding_box[2], bounding_box[3]),
    )

    projection = ccrs.Robinson(central_longitude=-90)

    fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={"projection": projection})

    ax.add_feature(cfeature.LAND, edgecolor="0.5", linewidth=0.5, zorder=-1)
    ax.add_feature(cfeature.OCEAN, color="lightblue")
    ax.set_extent([270, 330, 5, 55], crs=ccrs.PlateCarree())  # Atlantic (it=5+)
    # ax.set_extent([0, 360, -90, 90], crs=ccrs.PlateCarree())
    ax.coastlines(color="gray")
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
        draw_labels=True,
    )
    ax.set_title(f"{model_name}")

    # calculate and plot tracks
    mean_msl = mean_ds["msl"].values
    lat, lon = mean_ds.lat.values, mean_ds.lon.values
    pmsave = []
    col_inc = -1
    for i, amp in enumerate(amp_vec):
        tend_path = tendency_file = (
            output_dir / "auxiliary" / f"tendency_{model_name}_amp={amp}.nc"
        )
        moist_run = pert_params["moist_run"][i]
        if moist_run:
            col_inc += 1
        msl_time_series = (
            ds["msl"].isel(amplitude=i, moist_run=moist_run).values.squeeze()
        )
        latmin, lonmin, pmin = tc_vitals(msl_time_series - mean_msl, lat, lon)
        pmsave.append(pmin)

        # # remove noisy values by setting any pmin greater than -1 to the prev point
        # for j, p in enumerate(pmin):
        #     if j < 5:  # except for initial step(s)
        #         continue
        #     if pmin[j] > -1:
        #         latmin[j] = latmin[j - 1]
        #         lonmin[j] = lonmin[j - 1]
        # print(f"Amp = {amp}, i = {i}")
        # print(f"Latmin: {latmin}")
        # print(f"Lonmin: {lonmin}")

        latmin = np.insert(latmin, 0, np.array(15))
        lonmin = np.insert(lonmin, 0, np.array(320))
        if moist_run:
            ax.plot(
                lonmin,
                latmin,
                marker="o",
                label=rf"x {int(amp)}",
                alpha=1,
                color=cols[col_inc],
                linestyle="-",
                transform=ccrs.PlateCarree(),
            )
        else:
            print(f'not plotting dry run w/ {amp = }')

    plt.legend()
    gl.top_labels = False
    gl.right_labels = False

    plt.savefig(plot_dir / f"TC_tracks_{model_name}.png", dpi=300, bbox_inches="tight")

    timestep_hours = model_info.MODEL_TIME_STEP_HOURS[model_name]
    dstep = timestep_hours / 24
    days = np.arange(0, (n_timesteps * timestep_hours) / 24 + dstep, dstep)
    lw = 2
    fig, ax = plt.subplots()
    col_inc = -1
    for h in range(len(amp_vec)):
        moist_run = pert_params["moist_run"][h]
        if moist_run:
            col_inc += 1
            linestyle = "-"
        else:
            linestyle = "--"
        linecolor = cols[col_inc]
        ax.plot(
            days,
            pmsave[h],
            linewidth=lw,
            color=linecolor,
            linestyle=linestyle,
            label="x " + str(int(amp_vec[h])),
        )
        # if amp_vec[h] == 10:
        #     ax.plot(days[:2],pmsave_noq[:2],'k--',linewidth=lw,color=cols[h],label='x '+str(amps[h])+'_noq')

    xl = ax.get_xlim()
    ax.plot(xl, [0, 0], "k-", linewidth=1)
    ax.set_xlim(1, 12)
    plt.xlabel("time (days)", weight="bold")
    plt.ylabel("minimum MSLP anomaly (hPa)", weight="bold")
    plt.title(model_name)
    plt.setp(ax.spines.values(), linewidth=lw)
    plt.legend()
    plt.savefig(
        plot_dir / f"hurricane_timeseries_{model_name}.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    # # MSLP anomalies (global)
    # titles = [
    #     f"{model_name.upper()}: Q1000 Anomalies from JAS at t={t*model_info.MODEL_TIME_STEP_HOURS[model_name]} hours"
    #     for t in range(0, n_timesteps + 1)
    # ]
    # data = ds["q1000"].sel(amplitude=10, moist_run=1).isel(amplitude=0).squeeze() - mean_ds["q1000"].squeeze()
    # plot_var = f"q1000_global_{model_name}_10_q"
    # vis.create_and_plot_variable_gif(
    #     data=data,
    #     plot_var=plot_var,
    #     iter_var="lead_time",
    #     iter_vals=np.arange(0, n_timesteps + 1),
    #     plot_dir=plot_dir,
    #     units="?",
    #     cmap="Blues",
    #     titles=titles,
    #     keep_images=False,
    #     dpi=300,
    #     fps=2,
    #     # vlims=(0, 50),  # Set vlims for better visualization
    #     # extent=[270, 330, 5, 55],
    #     central_longitude=180.0,
    #     fig_size=(7.5, 3.5),
    #     adjust={
    #         "top": 0.97,
    #         "bottom": 0.01,
    #         "left": 0.09,
    #         "right": 0.87,
    #         "hspace": 0.0,
    #         "wspace": 0.0,
    #     },
    #     cbar_kwargs={
    #         "rotation": "horizontal",
    #         "y": -0.02,
    #         "horizontalalignment": "right",
    #         "labelpad": -34.5,
    #         "fontsize": 9,
    #     },
    # )

    # # print(f"Made {plot_var}.gif.")
    # # #
    # # ### End Tropical Cyclone Visualizations ###
    # # #
