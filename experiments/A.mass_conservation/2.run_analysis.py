import xarray as xr
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import calendar
from utils import inference_sfno
from utils_E2S import general
import math
from utils_E2S import vis
from matplotlib import colormaps

### Set up and parameter selection ########

# read configuration
config_path = Path(__file__).parent / "0.config.yaml"
config = general.read_config(config_path)

# get models and parameters from config
models = config["models"] 
plot_var = "msl" # choose either "msl" or "sp", can only use "sp" if models = ["sfno"] (other models don't output SP)
# vis options
cmap_str = "Dark2" # options here: matplotlib.org/stable/tutorials/colors/colormaps.html
day_interval_x_ticks = 2 # how many days between x-ticks on the plot
standardized_ylims = (1010, 1014) # y-limits for the plot, set to None to use the model output min/max

# set up directories
exp_dir = Path(config["experiment_dir"]) / config["experiment_name"]
plot_dir = exp_dir / "plots" # where to save plots

ic_dates = [dt.datetime.strptime(str_date, "%Y/%m/%d %H:%M") for str_date in config["ic_dates"]]
lead_times = np.arange(0, config["n_timesteps"] + 1) * 6  # in hours
n_ics = len(ic_dates)
n_timesteps = config["n_timesteps"]
# n_timesteps = 5 ###DEBUG
# lead_times = lead_times[:n_timesteps+1]###DEBUG
##############################################

### Loop through each model and plot the results ###
for model in models:
    ### Load model output data ###
    data_path = exp_dir / f"{model}_output.nc" # where output data is stored
    print(f"Loading data from {data_path}")
    ds = xr.open_dataset(data_path) / 100 # convert to hPa
    ds = ds.isel(lead_time=slice(0, n_timesteps+1)) # make sure we only get as much data as we need
    ds = ds.assign_attrs({"time units": "hours since start"})
    ds = ds.sortby("lat", ascending=True)
    ##############################################

    ### Plot the results ######################
    title = f"Simulated Pressure Trends\n{model.upper()}"
    save_title = f"{plot_var.lower()}_trends_{model}.png"
    ylab = "Lat-weighted SP (hPa)" if plot_var == "sp" else "Lat-weighted MSLP (hPa)"
    linewidth = 2
    fontsize = 24
    smallsize = 20
    fcst_linestyle = "solid" # see https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
    e5_linestyle = (0, (3, 1, 1, 1)) # densely dotted line
    cmap = colormaps.get_cmap(cmap_str)
    qual_colors = cmap(np.linspace(0, 1, n_ics))

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    for i, ic in enumerate(ic_dates):
        model_linedat = ds[f"MEAN_{plot_var}"].isel(init_time=i).squeeze()
        color = qual_colors[i]
        ax.plot(lead_times, model_linedat, color=color, linewidth=linewidth, label=f"Forecast init   {ic.strftime('%Y-%m-%d %Hz')}", linestyle=fcst_linestyle)
        
    ens_mean = ds[f"IC_MEAN_{plot_var}"].squeeze()
    ax.plot(lead_times, ens_mean, color="red", linewidth=2*linewidth, label="Mean of forecast lines", linestyle=fcst_linestyle)
       
    ax.set_xticks(lead_times[::4*day_interval_x_ticks], (lead_times[::4*day_interval_x_ticks]//(24)), fontsize=smallsize)
    if standardized_ylims:
        val_range = standardized_ylims
        print(f"Using standardized y-limits: {val_range}")
    else:
        val_range = (math.floor(ens_mean.min()), math.ceil(ens_mean.max()))
        print(f"Using model output min/max for y-limits ({model}): {val_range}")
    val_diff = val_range[1] - val_range[0]
    if val_diff < 2.0:
        ytick_interval = 0.5
    else:
        ytick_interval = 2.0
    yticks = np.arange(val_range[0], val_range[1]+ytick_interval, ytick_interval)
    ax.set_yticks(yticks, yticks, fontsize=smallsize)
    ax.set_xlabel("Simulation Time (days)", fontsize=fontsize)
    ax.set_ylabel(ylab, fontsize=fontsize)
    ax.set_xlim(xmin=-3, xmax=lead_times[-1]+3)
    fig.suptitle(title, fontsize=28)
    ax.grid()
    ax.set_facecolor("#FFFFFF")
    fig.tight_layout()
    plt.legend(fontsize=12, loc="upper left", ncols=2)
    plt.savefig(plot_dir / save_title, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {plot_dir / save_title}")
    ###########################################
    
    ### plot MSLP as a sanity check ###
    titles = [f"{model.upper()}: MSLP init {ic_dates[-1].strftime('%d-%m-%Y %Hz')} @ {(t/4):.2f} days lead time" for t in np.arange(0, n_timesteps+1)]
    data = ds["msl"].isel(init_time=-1).squeeze()  # select final init time
    gif_plot_var = f"MSLP_{model}"
    vis.create_and_plot_variable_gif(
        data=data,
        plot_var=gif_plot_var,
        iter_var="lead_time",
        iter_vals=np.arange(0, n_timesteps+1),
        plot_dir=plot_dir,
        units="hPa",
        cmap="bwr",
        titles=titles,
        keep_images=False,
        dpi=300,
        fps=8, 
        fig_size=(8, 4),
        vlims=(950, 1076),  # Set vlims for better visualization
        central_longitude=180.0,
        adjust = {
        "top": 0.93,
        "bottom": 0.03,
        "left": 0.09,
        "right": 0.87,
        "hspace": 0.0,
        "wspace": 0.0,
        },
        cbar_kwargs={
            "rotation": "horizontal",
            "y": -0.015,
            "horizontalalignment": "right",
            "labelpad": -29,
            "fontsize": 9
        },
    )
    print(f"Made {gif_plot_var}.gif.")
    