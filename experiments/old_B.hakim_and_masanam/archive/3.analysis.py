import xarray as xr
import datetime as dt
import numpy as np
from utils_E2S import vis
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import yaml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# choose model to visualize
models = ["sfno", "graphcast_oper", "pangu"] # full set is ["sfno", "graphcast_oper", "pangu"]

# read configuration
this_dir = Path(__file__).parent
config_path = this_dir / "0.config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    
# set up directories
exp_dir = Path(config["experiment_dir"]) / config["hm24_experiment"] / config["experiment_name"] # all data for experiment stored here
plot_dir = exp_dir / "plots" # save figures here
tendency_dir = Path(config["experiment_dir"]) / "tendencies" # tendency data stored here
IC_dir = Path(config["time_mean_IC_dir"])
IC_season = config["IC_season"]
heating_ds_path = exp_dir / f"heating.nc"
    
# convenience vars
n_timesteps = config["n_timesteps"]
n_timesteps = 40 # just for vis
lead_times_h = np.arange(0, 6*n_timesteps+1, 6)
g = 9.81 # m/s^2
hm24_exp = config["hm24_experiment"]

### create same visualizations for each model in list
print(f"Visualizing {hm24_exp} experiment for models: {models}")
for model in models:
    # load datasets
    IC_path = IC_dir / f"{IC_season}_ERA5_time_mean_{model}.nc"
    IC_ds = xr.open_dataset(IC_path).sortby("latitude", ascending=False)
    tendency_path = tendency_dir / f"{IC_season}_{model}_tendency.nc"
    model_output_path = exp_dir / f"{model}_output.nc"
    ds = xr.open_dataset(model_output_path)
    if heating_ds_path.exists():
        heating_ds = xr.open_dataset(heating_ds_path)
    tds = xr.open_dataset(tendency_path)
    mean_ds = xr.open_dataset(IC_path)
    ds["Z"] = ds["Z"] / (g) # convert to geopot. height
    mean_ds["Z"] = mean_ds["Z"] / (g) # convert to geopot. height

    print(f"Loaded data from {model}, beginning visualization.")
    
    #
    ### Begin Tropical Heating Visualizations ###
    #
    if hm24_exp == "tropical_heating":
        
        ### begin HM24 fig. 1
                
        projection = ccrs.Robinson(central_longitude=120.)
        fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(11*2,8.5*2),subplot_kw={'projection': projection}, layout='constrained')

        panel_label = ['(A)','(B)','(C)']
        plot_vec = False
        axi = -1
        lat, lon = ds["latitude"].values, ds["longitude"].values
        for it in [120,240,480]:
            axi+=1

            # h&m24 plot replication
            # _mean indicates the mean state
            # _pert indicates the perturbed run
            # _anom indicates the anomaly (perturbated run - mean state)
            ds500 = ds.sel(level=500, lead_time=it).squeeze()
            z500_mean = mean_ds["Z"].sel(level=500).squeeze().values
            z500_pert = ds500["Z"].values
            u500_pert = ds500["U"].values
            v500_pert = ds500["V"].values
            u500_mean = mean_ds["U"].sel(level=500).squeeze().values
            v500_mean = mean_ds["V"].sel(level=500).squeeze().values
            pzdat = z500_pert - z500_mean
            udat  = u500_pert - u500_mean
            vdat  = v500_pert - v500_mean
            basefield = z500_mean

            heating = heating_ds["T"].sel(level=500).squeeze().values
            if it == 0:
                dcint = .00001; ncint=5
            elif it == 120:
                dcint = .3; ncint=5
                vscale = 50 # vector scaling (counterintuitive:smaller=larger arrows)
            elif it == 240:
                dcint = 2; ncint=5        
                vscale = 100 # vector scaling (counterintuitive:smaller=larger arrows)
            else:
                dcint = 20; ncint=5
                vscale = 250 # vector scaling (counterintuitive:smaller=larger arrows)
            
            if plot_vec:
                # Plot vectors on the map
                latskip = 10
                lonskip = 10
                alpha = 0.75
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

            # plot heating
            cs = ax[axi].contour(lon,lat,heating,levels=[.05],colors='r',linestyles='dashed',linewidths=4,transform=ccrs.PlateCarree(),alpha=alpha)

            # colorize land
            ax[axi].add_feature(cfeature.LAND,edgecolor='0.5',linewidth=0.5,zorder=-1)

            gl = ax[axi].gridlines(crs=ccrs.PlateCarree(),linewidth=1.0,color='gray', alpha=0.5,linestyle='--', draw_labels=True)
            gl.top_labels = False
            if axi != 2:
                gl.bottom_labels = False
            gl.xlabels_left = True

            ax[axi].text(-0.02,0.02,panel_label[axi],transform=ax[axi].transAxes)

        fig.tight_layout()
        fname = f'heating_500z_{model}.pdf'
        plt.savefig(plot_dir / fname,dpi=100,bbox_inches='tight')
        print(f"Saved {fname} to {plot_dir}.")
        ### end HM24 fig. 1
        
        ### plot gifs
        # T500 anomalies
        titles = [f"{model.upper()}: $T_{{500}}$ Anomalies from {IC_season} Climatology at t={t*6} hours" for t in range(0, n_timesteps+1)]
        data = ds["T"].isel(ensemble=0).sel(level=500) - (mean_ds["T"].sel(level=500)).squeeze()
        plot_var = f"T500_anom_{model}"
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
        
        # Z500
        titles = [f"{model}: $Z_{{500}}$ at t={t*6} hours" for t in range(0, n_timesteps+1)]
        data = ds["Z"].isel(ensemble=0).sel(level=500)/(10)
        plot_var = f"Z500_{model}"
        vis.create_and_plot_variable_gif(
            data=data,
            plot_var=plot_var,
            iter_var="lead_time",
            iter_vals=np.arange(0, n_timesteps+1),
            plot_dir=plot_dir,
            units="dam",
            cmap="PRGn",
            titles=titles,
            keep_images=False,
            dpi=150,
            fps=2, 
            vlims=(450, 600),  # Set vlims for better visualization
            central_longitude=180.0,
        )
        print(f"Made {plot_var}.gif.")
        
        if model == "sfno":
            # show heating
            titles = ["Tropical heating term $f$ at levels 1000-200 hPa"]
            data = heating_ds["T"].squeeze().sel(level=[500], drop=False)
            plot_var = "heating_term"
            vis.create_and_plot_variable_gif(
                data=data,
                plot_var=plot_var,
                iter_var="level",
                iter_vals=[0],
                plot_dir=plot_dir,
                units="K/day",
                cmap="Reds",
                titles=titles,
                keep_images=False,
                dpi=300,
                fps=2, 
                vlims=(0, 0.1),  # Set vlims for better visualization
                central_longitude=180.0,
            )
            print(f"Made {plot_var}.gif.")
        
    #
    ### End Tropical Heating Visualizations ###
    #
    
    
    #
    ### Begin Extratropical Cyclone Visualizations ###
    #
    elif hm24_exp == "extratropical_cyclone":
        
        ### begin HM24 Fig. 3
        # 500Z plot
        plot_vec = True

        projection = ccrs.Robinson(central_longitude=-90.)
        fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(11*2,8.5*2),subplot_kw={'projection': projection}, layout='constrained')

        panel_label = ['(A)','(B)','(C)','(D)']

        axi = -1
        
        lat, lon = ds["latitude"].values, ds["longitude"].values
        for it in [0,48,72,96]:
            axi+=1

            # h&m24 plot replication
            # _mean indicates the mean state
            # _pert indicates the perturbed run
            # _anom indicates the anomaly (perturbated run - mean state)
            ds500 = ds.sel(level=500, lead_time=it).squeeze()
            z500_mean = mean_ds["Z"].sel(level=500).squeeze().values
            z500_pert = ds500["Z"].values
            u500_pert = ds500["U"].values
            v500_pert = ds500["V"].values
            u500_mean = mean_ds["U"].sel(level=500).squeeze().values
            v500_mean = mean_ds["V"].sel(level=500).squeeze().values
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

            ax[axi].text(130,20,f"{panel_label[axi]}:{it}hrs",transform=ccrs.PlateCarree())

        fig.tight_layout()
        plt.savefig(plot_dir/f'IVP_500_{model}.pdf',dpi=300,bbox_inches='tight')
        print(f"Saved IVP_500_{model}.pdf to {plot_dir}.")
        ### end HM24 Fig. 3

        ### make gifs
        # Z500 anomalies
        titles = [f"{model.upper()}: $Z_{{500}}$ Anomalies from {IC_season} Climatology at t={t*6} hours" for t in range(0, n_timesteps+1)]
        data = ds["Z"].isel(ensemble=0).sel(level=500) - (mean_ds["Z"].sel(level=500)).squeeze()
        plot_var = f"Z500_anom_{model}"
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

        # T500 anomalies
        titles = [f"{model.upper()}: $T_{{500}}$ Anomalies from {IC_season} Climatology at t={t*6} hours" for t in range(0, n_timesteps+1)]
        data = ds["T"].isel(ensemble=0).sel(level=500) - (mean_ds["T"].sel(level=500)).squeeze()
        plot_var = f"T500_anom_{model}"
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
    
    #
    ### Begin Geostrophic Adjustment Visualizations ###
    #
    elif hm24_exp == "geostrophic_adjustment":
        pass
    #
    ### End Geostrophic Adjustment Visualizations ###
    #
    
    #
    ### Begin Tropical Cyclone Visualizations ###
    #
    elif hm24_exp == "tropical_cyclone":
        pass
    #
    ### End Tropical Cyclone Visualizations ###
    #
    else:
        raise ValueError(f"Unknown experiment {hm24_exp} for model {model}. Please check the configuration file")
