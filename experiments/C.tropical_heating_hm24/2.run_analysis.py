from utils import general, vis
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
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

    # z1000
    nt = config["n_timesteps"]
    titles = [f"{model}: $Z_{{1000}}$ at t={t*6} hours" for t in range(0, nt+1)]
    data = ds["z1000"] / (9.8*10)
    plot_var = f"z1000_{model}"
    vis.create_and_plot_variable_gif(
        data=data,
        plot_var=plot_var,
        iter_var="lead_time",
        iter_vals=np.arange(0, nt+1),
        plot_dir=plot_dir,
        units="dam",
        cmap="PRGn",
        titles=titles,
        keep_images=False,
        dpi=300,
        fps=1, 
        vlims=(-50, 50),  # Set vlims for better visualization
        central_longitude=180.0,
    )
    print(f"Made {plot_var}.gif.")
    
    # z1000_anom
    nt = config["n_timesteps"]
    plot_var = f"z1000_anom_{model}"
    titles = [f"{model}: {plot_var} at t={t*6} hours" for t in range(0, nt+1)]
    data = (ds["z1000"] - ds["z1000"].isel(lead_time=0)) / (9.8*10)
    vis.create_and_plot_variable_gif(
        data=data,
        plot_var=plot_var,
        iter_var="lead_time",
        iter_vals=np.arange(0, nt+1),
        plot_dir=plot_dir,
        units="dam",
        cmap="PRGn",
        titles=titles,
        keep_images=False,
        dpi=300,
        fps=1, 
        vlims=(-5, 5),  # Set vlims for better visualization
        central_longitude=180.0,
    )
    print(f"Made {plot_var}.gif.")
    
    # heating w/ cartopy borders
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()})
    heating_ds["t1000"].isel(time=0).plot(ax=ax, cmap="RdBu", cbar_kwargs={"label": "Heating (K/day)"})
    ax.coastlines()
    ax.set_title(f"{model}: Heating")
    plt.savefig(plot_dir / f"heating_{model}.png", dpi=200)
    plt.close(fig)
    print(f"Made heating_{model}.png")
    
    # plot from paper
    ### begin HM24 fig. 1
            
    # projection = ccrs.Robinson(central_longitude=120.)
    # fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(11*2,8.5*2),subplot_kw={'projection': projection}, layout='constrained')

    # panel_label = ['(A)','(B)','(C)']
    # plot_vec = False
    # axi = -1
    # lat, lon = ds["lat"].values, ds["lon"].values
    # for it in [120,240,360]:
    #     axi+=1

    #     # h&m24 plot replication
    #     # _mean indicates the mean state
    #     # _pert indicates the perturbed run
    #     # _anom indicates the anomaly (perturbated run - mean state)
    #     ds500 = ds.sel(lead_time=it).squeeze()
    #     z1000_mean = ds["z1000"].isel(lead_time=0).squeeze().values
    #     z1000_pert = ds500["z1000"].values
    #     u500_pert = ds500["u500"].values
    #     v500_pert = ds500["v500"].values
    #     u500_mean = ds["u500"].isel(lead_time=0).squeeze().values
    #     v500_mean = ds["v500"].isel(lead_time=0).squeeze().values
    #     pzdat = z1000_pert - z1000_mean
    #     udat  = u500_pert - u500_mean
    #     vdat  = v500_pert - v500_mean
    #     basefield = z1000_mean

    #     heating = heating_ds["t500"].isel(time=0).squeeze().values
    #     if it == 0:
    #         dcint = .000001; ncint=1
    #     elif it == 120:
    #         dcint = 1; ncint=1
    #         vscale = 50 # vector scaling (counterintuitive:smaller=larger arrows)
    #     elif it == 240:
    #         dcint = 2; ncint=1        
    #         vscale = 100 # vector scaling (counterintuitive:smaller=larger arrows)
    #     else:
    #         dcint = 20; ncint=1
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
    #     print(f"Time: {it} hours")
    #     print(f"Negative intervals: {cints_neg}")
    #     print(f"Positive intervals: {cints_pos}")
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

    # fig.tight_layout()
    # fname = f'heating_500z_{model}.pdf'
    # plt.savefig(plot_dir / fname,dpi=100,bbox_inches='tight')
    # print(f"Saved {fname} to {plot_dir}.")
    #     ### end HM24 fig. 1