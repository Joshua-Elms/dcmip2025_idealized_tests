import xarray as xr
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
import datetime as dt
import calendar
from utils import inference_sfno
import math
from utils import vis
from matplotlib import colormaps

### Set up and parameter selection ########

# read configuration
this_dir = Path(__file__).parent
config_path = this_dir / "0.config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    
# choose model & var to visualize
models = ["sfno", "graphcast_oper", "pangu"] # full set is ["sfno", "graphcast_oper", "pangu"]
plot_var = "MSL" # choose either "MSL" or "SP", can only use "SP" if models = ["sfno"] (other models don't output SP)
# vis options
cmap_str = "Dark2" # options here: matplotlib.org/stable/tutorials/colors/colormaps.html
day_interval_x_ticks = 15 # how many days between x-ticks on the plot
standardized_ylims = (1010, 1020) # y-limits for the plot, set to None to use the model output min/max
plot_ERA5_raw = False # whether to plot the raw ERA5 data (not averaged over initial conditions), in which case data must be re-downloaded instead of used from cache

# set up directories
exp_dir = Path(config["experiment_dir"]) / config["experiment_name"] # all data for experiment stored here
era5_pressure_path = exp_dir / "era5_pressure.nc" # where to save cached ERA5 data
plot_dir = exp_dir / "plots" # where to save plots
e5_base = "/glade/campaign/collections/rda/data/ds633.0/"

ic_dates = [dt.datetime.strptime(str_date, "%Y/%m/%d %H:%M") for str_date in config["ic_dates"]]
lead_times = np.arange(0, config["n_timesteps"] + 1) * 6  # in hours
n_ics = len(ic_dates)
n_timesteps = config["n_timesteps"]
# n_timesteps = 5 ###DEBUG
# lead_times = lead_times[:n_timesteps+1]###DEBUG
##############################################

### Loading the ERA5 data for comparison ###

# check to see if the data is already cached
cached = True
if plot_ERA5_raw:
    print("Plotting raw ERA5 data, will not use cached data.")
    cached = False
if era5_pressure_path.exists():
    e5_pds = xr.open_dataset(era5_pressure_path)
    # check whether the cached data has at least as many timesteps as the model output
    if len(e5_pds.init_time) < n_ics:
        print(f"Cached data has only {len(e5_pds.init_time)} timesteps, will re-download.")
        cached = False
        
    # check whether the cached data has the same initial conditions as the model output
    for ic_date in ic_dates:
        # check whether all dates are in the cached data
        if np.datetime64(ic_date) not in e5_pds.init_time.values:
            print(f"Missing data for {ic_date}, will re-download.")
            cached = False
            break
        
    print(f"Loaded cached data from {era5_pressure_path} with {len(e5_pds.init_time)} timesteps.")
    
else:
    cached = False
    
if not cached:
    print(f"Loading ERA5 pressure data from {e5_base}")

    # path to the ERA5 pressure data
    sp_template = f"{e5_base}/e5.oper.an.sfc/{{year:04}}{{month:02}}/e5.oper.an.sfc.128_134_sp.ll025sc.{{year:04}}{{month:02}}0100_{{year:04}}{{month:02}}{{dayend:02}}23.nc"
    msl_template = f"{e5_base}/e5.oper.an.sfc/{{year:04}}{{month:02}}/e5.oper.an.sfc.128_151_msl.ll025sc.{{year:04}}{{month:02}}0100_{{year:04}}{{month:02}}{{dayend:02}}23.nc"
    pressure_dict = {}
    for ic_date in ic_dates:
        keep_times = ic_date + dt.timedelta(hours=6) * np.arange(n_timesteps+1)
        pressure_files = []
        # necessary in case the user requests inference that spans multiple months
        year_months = set((t.year, t.month) for t in keep_times)
        for year, month in year_months:
            dayend = calendar.monthrange(year, month)[1]
            sp_file = sp_template.format(year=year, month=month, dayend=dayend)
            msl_file = msl_template.format(year=year, month=month, dayend=dayend)
            pressure_files.append(sp_file)
            pressure_files.append(msl_file)

        pds = xr.open_mfdataset(pressure_files, parallel=True)
        pds = pds.sel(time=keep_times).squeeze()
        sp_latmean = inference_sfno.latitude_weighted_mean(pds["SP"], pds["latitude"]) / 100 # convert to hPa
        msl_latmean = inference_sfno.latitude_weighted_mean(pds["MSL"], pds["latitude"]) / 100 # convert to hPa
        pressure_dict[ic_date] = xr.Dataset(
            data_vars={
                "SP": (["lead_time"], sp_latmean.values),
                "MSL": (["lead_time"], msl_latmean.values),
            },
            coords={
                "init_time": ic_date,
                "lead_time": lead_times,
            }
        )
        print(f"Loaded pressure for {ic_date} with shape {pds['SP'].shape}")

    print(f"Caching the ERA5 data for {len(ic_dates)} start dates.")
    # combine the data into a single xarray dataset
    e5_pds = xr.concat([pressure_dict[ic] for ic in ic_dates], dim="init_time")
    e5_pds["SP_mean"] = e5_pds["SP"].mean(dim="init_time")
    e5_pds["MSL_mean"] = e5_pds["MSL"].mean(dim="init_time")

    # cache the data for quicker visualizations in the future
    e5_pds.to_netcdf(era5_pressure_path, mode="w", format="NETCDF4", engine="netcdf4")
############################################

### Loop through each model and plot the results ###
for model in models:
    ### Load model output data ###
    data_path = exp_dir / f"{model}_output.nc" # where output data is stored
    print(f"Loading data from {data_path}")
    ds = xr.open_dataset(data_path)
    ds = ds.isel(time=slice(0, n_timesteps+1)) # make sure we only get as much data as we need
    ds = ds.assign_attrs({"time units": "hours since start"})
    ##############################################

    ### Plot the results ######################
    title = f"Simulated Pressure Trends\n{model.upper()} vs. ERA5"
    save_title = f"{plot_var.lower()}_trends_{model}_era5.png"
    ylab = "Lat-weighted SP (hPa)" if plot_var == "SP" else "Lat-weighted MSLP (hPa)"
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
        e5_linedat = e5_pds[plot_var].isel(init_time=i).squeeze()
        color = qual_colors[i]
        ax.plot(lead_times, model_linedat, color=color, linewidth=linewidth, label=f"Forecast init   {ic.strftime('%Y-%m-%d %Hz')}", linestyle=fcst_linestyle)
        ax.plot(lead_times, e5_linedat, color=color, linewidth=linewidth, label=f"ERA5 starting {ic.strftime('%Y-%m-%d %Hz')}", linestyle=e5_linestyle)
        
    ens_mean = ds[f"IC_MEAN_{plot_var}"].squeeze()
    ax.plot(lead_times, ens_mean, color="red", linewidth=2*linewidth, label="Mean of forecast lines", linestyle=fcst_linestyle)
    ax.plot(lead_times, e5_pds[f"{plot_var}_mean"], color="red", linewidth=2*linewidth, label="Mean of ERA5 lines", linestyle=e5_linestyle)
       
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
    data = ds["MSL"].isel(ensemble=0, init_time=-1).squeeze()  # select final init time
    gif_plot_var = f"MSLP_{model}"
    vis.create_and_plot_variable_gif(
        data=data,
        plot_var=gif_plot_var,
        iter_var="time",
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
    
if plot_ERA5_raw:
   
    # create titles for the GIF
    titles = [f"ERA5 MSLP @ {ic_dates[-1].strftime('%d-%m-%Y %Hz')} + {(t/4):.2f} days" for t in np.arange(0, n_timesteps+1)]
    data = (pds["MSL"] / 100).sortby("latitude", ascending=True) # select last init time because it's the last var that the for loop touched
    gif_plot_var = f"MSLP_ERA5"
    vis.create_and_plot_variable_gif(
        data=data,
        plot_var=gif_plot_var,
        iter_var="time",
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
        cbar_kwargs = {
            "rotation": "horizontal",
            "y": -0.015,
            "horizontalalignment": "right",
            "labelpad": -29,
            "fontsize": 9
        },
    )
    print(f"Made {gif_plot_var}.gif.")