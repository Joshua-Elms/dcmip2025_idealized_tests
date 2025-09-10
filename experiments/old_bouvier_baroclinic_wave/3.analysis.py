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

# set up paths
this_dir = Path(__file__).parent

# read configuration
config_path = this_dir / "0.config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    
exp_dir = Path(config["experiment_dir"]) / config["experiment_name"]
plot_dir = exp_dir / "plots" # save figures here
data_path = exp_dir / "output.nc" # where output from inference was saved
tendency_reversion = config["inference_parameters"]["tendency_reversion"]
perturbed = config["perturbation_parameters"]["enabled"]

# convenience vars
n_timesteps = config["inference_parameters"]["n_steps"]
lead_times_h = np.arange(0, 6*n_timesteps+1, 6)
g = 9.81 # m/s^2

# load datasets
ds = xr.open_dataset(data_path)
mean_ds = xr.open_dataset(exp_dir/"ic_nc/ic_zt0=288.nc").squeeze().rename({"lat": "latitude", "lon": "longitude"})
perturbed = config["perturbation_parameters"]["enabled"]
if perturbed:
    ipert_ds = xr.open_dataset(exp_dir/"ic_nc/initial_perturbation.nc").sortby("latitude", ascending=False)

# make pretty plots
# titles = [f"VAR_2T at t={t*6} hours" for t in range(n_timesteps+1)]
# vis.create_and_plot_variable_gif(
#     data=ds["VAR_2T"].isel(ensemble=0),
#     plot_var="VAR_2T",
#     iter_var="lead_time",
#     iter_vals=np.arange(0, n_timesteps+1),
#     plot_dir=plot_dir,
#     units="degrees K",
#     cmap="magma",
#     titles=titles,
#     keep_images=False,
#     dpi=300,
#     fps=8,
# )

# print(f"Made VAR_2T.gif.")

# titles = [f"$Z_{{500}}$ at t={t*6} hours" for t in range(n_timesteps+1)]
# vis.create_and_plot_variable_gif(
#     data=ds["Z"].isel(ensemble=0).sel(level=500)/(10*g),
#     plot_var="Z500",
#     iter_var="lead_time",
#     iter_vals=np.arange(0, n_timesteps+1),
#     plot_dir=plot_dir,
#     units="dam",
#     cmap="coolwarm",
#     titles=titles,
#     keep_images=False,
#     dpi=300,
#     fps=8,
# )

# print(f"Made Z500.gif.")
lim = 16
# titles = [f"$Z_{{500}}$ at t={t*6} hours (TR={tendency_reversion}, P={perturbed})" for t in range(0, lim, 1)]
# data = ds["Z"].isel(ensemble=0, zt0=0).sel(level=500)/(10*g)
# z500_mid = data.isel(lead_time=0).mean(dim=["latitude", "longitude"]).values
# vis.create_and_plot_variable_gif(
#     data=data.isel(lead_time=slice(0, lim)),
#     plot_var="Z500",
#     iter_var="lead_time",
#     iter_vals=np.arange(0, lim, 1),
#     plot_dir=plot_dir,
#     units="dam",
#     cmap="bwr",
#     titles=titles,
#     keep_images=False,
#     dpi=300,
#     fps=0.5, 
#     vlims=(z500_mid-30, z500_mid+30),
# )

# print(f"Made Z500.gif.")

# mslp_mid = 1013.25
# titles = [f"MSLP at t={t*6} hours (TR={tendency_reversion}, P={perturbed})" for t in range(0, lim, 1)]
# data = ds["MSL"].isel(zt0=0, ensemble=0) / 100 # convert to hPa
# vis.create_and_plot_variable_gif(
#     data=data.isel(lead_time=slice(0, lim)),
#     plot_var="MSLP",
#     iter_var="lead_time",
#     iter_vals=np.arange(0, lim, 1),
#     plot_dir=plot_dir,
#     units="hPa",
#     cmap="bwr",
#     titles=titles,
#     keep_images=False,
#     dpi=300,
#     fps=0.5,
#     vlims=(mslp_mid-8, mslp_mid+8),
# )

# print(f"Made MSLP.gif.")

titles = [f"$Z_{{500}}$ anomalies at t={t*6} hours (TR={tendency_reversion}, P={perturbed})" for t in range(0, lim, 1)]
z500_anom = ds["Z"].isel(ensemble=0, zt0=0).sel(level=500)/(g) - mean_ds["Z"].sel(level=500) / (g)
vis.create_and_plot_variable_gif(
    data=z500_anom.isel(lead_time=slice(0, lim)),
    plot_var="Z500_anom",
    iter_var="lead_time",
    iter_vals=np.arange(0, lim, 1),
    plot_dir=plot_dir,
    units="m",
    cmap="bwr",
    titles=titles,
    keep_images=False,
    dpi=300,
    fps=0.5, 
    vlims=(-5, 5),
)

print(f"Made Z500_anom.gif.")

# plot Z anoms at diff levels
choose_t = 12
titles = [f"$Z_{{{lev}}}$ anomalies at t={choose_t*6} hours (TR={tendency_reversion}, P={perturbed})" for lev in ds.level.values]
data = ds["Z"].isel(ensemble=0, zt0=0, lead_time=choose_t)/(g) - mean_ds["Z"] / (g)
vis.create_and_plot_variable_gif(
    data=data,
    plot_var="Z_anom_levs",
    iter_var="level",
    iter_vals=[i for i, v in enumerate(ds.level.values)],
    plot_dir=plot_dir,
    units="m",
    cmap="bwr",
    titles=titles,
    keep_images=False,
    dpi=300,
    fps=0.5,
    vlims=(-10, 10),
)

print(f"Made Z_anom_levs.gif.")

titles = [f"MSLP anomalies at t={t*6} hours (TR={tendency_reversion}, P={perturbed})" for t in range(0, lim, 1)]
msl_anom = ds["MSL"].isel(zt0=0, ensemble=0) / 100 - mean_ds["MSL"] / 100 # convert to hPa
vis.create_and_plot_variable_gif(
    data=msl_anom.isel(lead_time=slice(0, lim)),
    plot_var="MSLP_anom",
    iter_var="lead_time",
    iter_vals=np.arange(0, lim, 1),
    plot_dir=plot_dir,
    units="hPa",
    cmap="bwr",
    titles=titles,
    keep_images=False,
    dpi=300,
    fps=0.5,
    vlims=(-2, 2),
)

print(f"Made MSLP_anom.gif.")

if perturbed:
    # show initial perturbation
    titles = [f"Initial U-Wind Perturbation (all levels)"]
    vis.create_and_plot_variable_gif(
        data=ipert_ds["U"].isel(ensemble=0).sel(level=500),
        plot_var="U_perturbation",
        iter_var="time",
        iter_vals=[0],
        plot_dir=plot_dir,  
        units="m/s",
        cmap="Greens",
        titles=titles,
        keep_images=True,
        dpi=300,
        fps=1,
        vlims=(0, 1),
    )
        
    print("Made U_perturbation.gif.")

# # debug tds
# titles = ["DJF VAR_2T Tendency (SFNO)"]
# tds["VAR_d2T_dt"] = tds["VAR_2T"] / (6)
# vis.create_and_plot_variable_gif(
#     data=tds["VAR_d2T_dt"],
#     plot_var="VAR_2T_tendency",
#     iter_var="ensemble",
#     iter_vals=[0],
#     plot_dir=plot_dir,
#     units="K hr$^{-1}$",
#     cmap="magma",
#     titles=titles,
#     keep_images=False,
#     dpi=300,
#     fps=1,
# )
# print(f"Made VAR_2T_tendency.gif.")


# titles = ["Heating Term $f$: $T_{500}$ Perturbation"]
# vis.create_and_plot_variable_gif(
#     data=heating_ds["T"].sel(level=500).isel(time=0),
#     plot_var="T500_heating",
#     iter_var="ensemble",
#     iter_vals=[0],
#     plot_dir=plot_dir,
#     units="$\Delta$ degrees K",
#     cmap="Reds",
#     titles=titles,
#     keep_images=False,
#     dpi=300,
#     fps=1,
# )
# print(f"Made T500_Heating.gif.")

# projection = ccrs.Robinson(central_longitude=120.)
# fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(11*2,8.5*2),subplot_kw={'projection': projection}, layout='constrained')

# panel_label = ['(A)','(B)','(C)']
# plot_vec = False
# axi = -1
# ds["Z"] = ds["Z"] / (g) # convert to geopot. height
# mean_ds["Z"] = mean_ds["Z"] / (g) # convert to geopot. height
# lat, lon = ds["latitude"].values, ds["longitude"].values
# for it in [120,240,480]:
#     axi+=1

#     # h&m24 plot replication
#     # _mean indicates the mean state
#     # _pert indicates the perturbed run
#     # _anom indicates the anomaly (perturbated run - mean state)
#     ds500 = ds.sel(level=500, lead_time=it).squeeze()
#     z500_mean = mean_ds["Z"].sel(level=500).squeeze().values
#     z500_pert = ds500["Z"].values
#     u500_pert = ds500["U"].values
#     v500_pert = ds500["V"].values
#     u500_mean = mean_ds["U"].sel(level=500).squeeze().values
#     v500_mean = mean_ds["V"].sel(level=500).squeeze().values
#     pzdat = z500_pert - z500_mean
#     udat  = u500_pert - u500_mean
#     vdat  = v500_pert - v500_mean
#     basefield = z500_mean
    
#     heating = heating_ds["T"].sel(level=500).squeeze().values
    
#     if it == 0:
#         dcint = .00001; ncint=5
#     elif it == 120:
#         dcint = .3; ncint=5
#         vscale = 50 # vector scaling (counterintuitive:smaller=larger arrows)
#     elif it == 240:
#         dcint = 2; ncint=5        
#         vscale = 100 # vector scaling (counterintuitive:smaller=larger arrows)
#     else:
#         dcint = 20; ncint=5
#         vscale = 250 # vector scaling (counterintuitive:smaller=larger arrows)
    
#     if plot_vec:
#         # Plot vectors on the map
#         latskip = 10
#         lonskip = 10
#         alpha = 0.75
#         col = 'g'
#         cs = ax[axi].quiver(lon[::lonskip],lat[::latskip],udat[::latskip,::lonskip],vdat[::latskip,::lonskip],transform=ccrs.PlateCarree(),scale=vscale,color=col,alpha=alpha)
#         qk = ax[axi].quiverkey(cs, 0.65, 0.01, 10., r'$10~ m/s$', labelpos='E',coordinates='figure',color=col)

#     # mean state or full field
#     alpha = 1.0
#     cints = np.arange(4800,6000,60.)
#     cs = ax[axi].contour(lon,lat,basefield,levels=cints,colors='0.5',transform=ccrs.PlateCarree(),alpha=alpha)
#     # perturbations
#     alpha = 1.0
#     cints = list(np.arange(-ncint*dcint,-dcint+.001,dcint))+list(np.arange(dcint,ncint*dcint+.001,dcint))
#     cints_neg = list(np.arange(-ncint*dcint,-dcint+.001,dcint))
#     cints_pos = list(np.arange(dcint,ncint*dcint+.001,dcint))
#     lw = 2.
#     cs = ax[axi].contour(lon,lat,pzdat,levels=cints_neg,colors='b',linestyles='solid',linewidths=lw,transform=ccrs.PlateCarree(),alpha=alpha)
#     cs = ax[axi].contour(lon,lat,pzdat,levels=cints_pos,colors='r',linestyles='solid',linewidths=lw,transform=ccrs.PlateCarree(),alpha=alpha)

#     # plot heating
#     cs = ax[axi].contour(lon,lat,heating,levels=[.05],colors='r',linestyles='dashed',linewidths=4,transform=ccrs.PlateCarree(),alpha=alpha)

#     # colorize land
#     ax[axi].add_feature(cfeature.LAND,edgecolor='0.5',linewidth=0.5,zorder=-1)

#     gl = ax[axi].gridlines(crs=ccrs.PlateCarree(),linewidth=1.0,color='gray', alpha=0.5,linestyle='--', draw_labels=True)
#     gl.top_labels = False
#     if axi != 2:
#         gl.bottom_labels = False
#     gl.xlabels_left = True

#     ax[axi].text(-0.02,0.02,panel_label[axi],transform=ax[axi].transAxes)

#     fig.tight_layout()
#     plt.savefig(plot_dir / 'heating_500z_day20.pdf',dpi=300,bbox_inches='tight')
