from utils import vis, general, model_info
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

print("Visualizing results of HM24 ETC Experiment")

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
g = 9.81 # m/s^2

for model_name in models:
    print(f"Visualizing {model_name}")
    IC_path = Path(IC_params["HM24_IC_dir"]) / f"{model_name}.nc"
    perturbation_path = Path(pert_params["perturbation_dir"]) / f"{season}_40N_150E_z-regression_{model_name}.nc"
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
    ### Begin Extratropical Cyclone Visualizations ###
    #
    if ds.sizes["lead_time"] >= 16:
        ### begin HM24 Fig. 3
        # 500Z plot
        plot_vec = True

        projection = ccrs.Robinson(central_longitude=-90.)
        fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(11*2,8.5*2),subplot_kw={'projection': projection}, layout='constrained')

        panel_label = ['(A)','(B)','(C)','(D)']

        axi = -1
        
        lat, lon = ds["lat"].values, ds["lon"].values
        for it in [0,48,72,96]: # lead times in hours
            axi+=1

            # h&m24 plot replication
            # _mean indicates the mean state
            # _pert indicates the perturbed run
            # _anom indicates the anomaly (perturbated run - mean state)

            z500_mean = mean_ds["z500"].squeeze().values
            u500_mean = mean_ds["u500"].squeeze().values
            v500_mean = mean_ds["v500"].squeeze().values
            z500_pert = ds["z500"].sel(lead_time=np.timedelta64(it, "h")).squeeze().values
            u500_pert = ds["u500"].sel(lead_time=np.timedelta64(it, "h")).squeeze().values
            v500_pert = ds["v500"].sel(lead_time=np.timedelta64(it, "h")).squeeze().values
            pzdat = z500_pert - z500_mean
            udat  = u500_pert - u500_mean
            vdat  = v500_pert - v500_mean
            basefield = z500_mean

            dcint = 20; ncint=5
            vscale = 250 # vector scaling (counterintuitive:smaller=larger arrows)

            if plot_vec:
                # Plot vectors on the map
                latskip = 10
                lonskip = 10
                alpha = 1.0
                col = 'g'
                cs = ax[axi].quiver(lon[::lonskip],lat[::latskip],udat[::latskip,::lonskip],vdat[::latskip,::lonskip],transform=ccrs.PlateCarree(),scale=vscale,color=col,alpha=alpha)
                qk = ax[axi].quiverkey(cs, 0.65, 0.01, 10., r'$10~ m/s$', labelpos='E',coordinates='figure',color=col)

            # mean state or full field
            alpha = 1.0
            cints = np.arange(4800,6000,60.)
            cs = ax[axi].contour(lon,lat,basefield,levels=cints,colors='0.5',transform=ccrs.PlateCarree(),alpha=alpha)
            # perturbations
            alpha = 1.0
            cints = list(np.arange(-ncint*dcint,-dcint+.001,dcint))+list(np.arange(dcint,ncint*dcint+.001,dcint))
            cints_neg = list(np.arange(-ncint*dcint,-dcint+.001,dcint))
            cints_pos = list(np.arange(dcint,ncint*dcint+.001,dcint))
            lw = 2.
            cs = ax[axi].contour(lon,lat,pzdat,levels=cints_neg,colors='b',linestyles='solid',linewidths=lw,transform=ccrs.PlateCarree(),alpha=alpha)
            cs = ax[axi].contour(lon,lat,pzdat,levels=cints_pos,colors='r',linestyles='solid',linewidths=lw,transform=ccrs.PlateCarree(),alpha=alpha)

            # colorize land
            ax[axi].add_feature(cfeature.LAND,edgecolor='0.5',linewidth=0.5,zorder=-1)

            # gridlines
            gl = ax[axi].gridlines(crs=ccrs.PlateCarree(),linewidth=1.0,color='gray', alpha=0.5,linestyle='--', draw_labels=True)

            ax[axi].set_extent([140, 260, 20, 70],crs=ccrs.PlateCarree()) # Pacific

            ax[axi].text(130,20,f"{panel_label[axi]}:{it*6}hrs",transform=ccrs.PlateCarree())

        fig.tight_layout()
        plt.savefig(plot_dir/f'IVP_500_{model_name}.pdf',dpi=300,bbox_inches='tight')
        print(f"Saved IVP_500_{model_name}.pdf to {plot_dir}.")
        ### end HM24 Fig. 3
    
    if model_name == "Pangu24":
        continue # diff timestep, skip rest of analysis for now

    ### make gifs
    # Z500 anomalies (global)
    titles = [f"{model_name.upper()}: $Z_{{500}}$ Anomalies from {season} Climatology at t={t*6} hours" for t in range(0, n_timesteps+1)]
    data = ds["z500"].squeeze() - (mean_ds["z500"]).squeeze()
    plot_var = f"z500_anom_global_{model_name}"
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
        vlims=(-150, 150),  # Set vlims for better visualization
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
    
    # Z500 anomalies (regional)
    titles = [f"{model_name.upper()}: $Z_{{500}}$ Anomalies from {season} Climatology at t={t*6} hours" for t in range(0, n_timesteps+1)]
    data = ds["z500"].squeeze() - (mean_ds["z500"]).squeeze()
    plot_var = f"z500_anom_regional_{model_name}"
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
        vlims=(-150, 150),  # Set vlims for better visualization
        central_longitude=180.0,
        extent=[120, 280, 0, 70],
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
    titles = [f"{model_name.upper()}: MSLP at t={t*6} hours" for t in range(0, n_timesteps+1)]
    data = ds["msl"].squeeze()
    plot_var = f"msl_global_{model_name}"
    vis.create_and_plot_variable_gif(
        data=data,
        plot_var=plot_var,
        iter_var="lead_time",
        iter_vals=np.arange(0, n_timesteps+1),
        plot_dir=plot_dir,
        units="m",
        cmap="viridis",
        titles=titles,
        keep_images=False,
        dpi=300,
        fps=2, 
        vlims=(950, 1030),  # Set vlims for better visualization
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

    # T500 anomalies (regional)
    titles = [f"{model_name.upper()}: $T_{{500}}$ Anomalies from {season} Climatology at t={t*6} hours" for t in range(0, n_timesteps+1)]
    data = ds["t500"].squeeze() - (mean_ds["t500"]).squeeze()
    plot_var = f"t500_anom_regional_{model_name}"
    vis.create_and_plot_variable_gif(
        data=data,
        plot_var=plot_var,
        iter_var="lead_time",
        iter_vals=np.arange(0, n_timesteps+1),
        plot_dir=plot_dir,
        units="K",
        cmap="bwr",
        titles=titles,
        keep_images=False,
        dpi=300,
        fps=2, 
        vlims=(-15, 15),  # Set vlims for better visualization
        central_longitude=180.0,
        extent=[120, 280, 0, 70],
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
            "labelpad": -26.5,
            "fontsize": 9
        },
    )

    print(f"Made {plot_var}.gif.")
    
    #
    ### End Extratropical Cyclone Visualizations ###
    #
    