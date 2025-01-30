
import datetime
import calendar
#import logging
import dask
import xarray as xr
import numpy as np
import torch


# Note: the 73 channel SFNO  model uses the following fields
# 
# channel = "u10m", "v10m", "u100m", "v100m", "t2m", "sp", "msl", "tcwv", 
#    "u50", "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", 
#    "u700", "u850", "u925", "u1000", "v50", "v100", "v150", "v200", "v250", 
#    "v300", "v400", "v500", "v600", "v700", "v850", "v925", "v1000", "z50", 
#    "z100", "z150", "z200", "z250", "z300", "z400", "z500", "z600", "z700", 
#    "z850", "z925", "z1000", "t50", "t100", "t150", "t200", "t250", "t300", 
#    "t400", "t500", "t600", "t700", "t850", "t925", "t1000", "r50", "r100", 
#    "r150", "r200", "r250", "r300", "r400", "r500", "r600", "r700", "r850", 
#    "r925", "r1000" ;

sfno_levels = [50,100,150,200,250,300,400,500,600,700,850,925,1000]
sfno_3dvars = ["U", "V", "Z", "T", "R"]

def pack_sfno_state(
        ds : xr.Dataset,
        device : str = "cpu",
        ) -> torch.Tensor:
    """ Takes an xarray dataset with the necessary fields and packs it into a tensor for the SFNO model. 

    input:
    ------

    ds : xr.Dataset
        an xarray dataset with the following fields:
        - U : (time, level, latitude, longitude) float32
        - V : (time, level, latitude, longitude) float32
        - Z : (time, level, latitude, longitude) float32
        - T : (time, level, latitude, longitude) float32
        - R : (time, level, latitude, longitude) float32
        - VAR_10U : (time, latitude, longitude) float32
        - VAR_10V : (time, latitude, longitude) float32
        - VAR_100U : (time, latitude, longitude) float32
        - VAR_100V : (time, latitude, longitude) float32
        - VAR_2T : (time, latitude, longitude) float32
        - SP : (time, latitude, longitude) float32
        - MSL : (time, latitude, longitude) float32
        - TCW : (time, latitude, longitude) float32
        - T2M : (time, latitude, longitude) float32

        Note that the following specific levels are expected to be available: 
        [50,100,150,200,250,300,400,500,600,700,850,925,1000] 

    device : str
        the device to put the tensor on

    output:
    -------

    x : torch.Tensor
        a tensor with dimensions (1, 1, 73, 721, 1440) containing the packed data
    """


    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        # concatenate the 3d variables along a new axis
        x3d = None
        for var in sfno_3dvars:
            for lev in sfno_levels:
                if x3d is None:
                    x3d = ds[var].sel(level=lev).drop_vars('level').squeeze()
                else:
                    x3d = xr.concat((x3d, ds[var].sel(level=lev).drop_vars('level').squeeze()), dim = "n")

        # concatenate the 2d variables
        x2d = ds["VAR_10U"]
        x2d = xr.concat((x2d, ds["VAR_10V"]), dim = "n")
        x2d = xr.concat((x2d, ds["VAR_100U"]), dim = "n")
        x2d = xr.concat((x2d, ds["VAR_100V"]), dim = "n")
        x2d = xr.concat((x2d, ds["VAR_2T"]), dim = "n")
        x2d = xr.concat((x2d, ds["SP"]), dim = "n")
        x2d = xr.concat((x2d, ds["MSL"]), dim = "n")
        x2d = xr.concat((x2d, ds["TCW"]), dim = "n")

        # concatenate the 2d and 3d variables
        x = xr.concat((x2d, x3d), dim = "n")

    x = torch.from_numpy(x.values).to(device=device)

    # add ensemble and time dimension
    x = x[None, None, ...]

    return x

def read_sfno_vars_from_era5_rda(
        time : datetime.datetime,
        e5_base : str = "/glade/campaign/collections/rda/data/ds633.0/",
        ) -> xr.Dataset:
    """ Fetches the ERA5 data for a specific time necessary for running the SFNO model. 
    
    input:
    ------

    time : datetime.datetime
        the time to fetch the data for

    e5_base : str
        the base path to the ERA5 data

    output:
    -------

    ds : xr.Dataset
        An xarray dataset with the following fields:
        - U : (time, level, latitude, longitude) float32
        - V : (time, level, latitude, longitude) float32
        - Z : (time, level, latitude, longitude) float32
        - T : (time, level, latitude, longitude) float32
        - R : (time, level, latitude, longitude) float32
        - VAR_10U : (time, latitude, longitude) float32
        - VAR_10V : (time, latitude, longitude) float32
        - VAR_100U : (time, latitude, longitude) float32
        - VAR_100V : (time, latitude, longitude) float32
        - VAR_2T : (time, latitude, longitude) float32
        - SP : (time, latitude, longitude) float32
        - MSL : (time, latitude, longitude) float32
        - TCW : (time, latitude, longitude) float32
        - T2M : (time, latitude, longitude) float32

        The levels specific to SFNO are selected: [50,100,150,200,250,300,400,500,600,700,850,925,1000]
    
    """

   # set the file name templates
    u_template = f"{e5_base}/e5.oper.an.pl/{{year:04}}{{month:02}}/e5.oper.an.pl.128_131_u.ll025uv.{{year:04}}{{month:02}}{{day:02}}00_{{year:04}}{{month:02}}{{day:02}}23.nc"
    v_template = f"{e5_base}/e5.oper.an.pl/{{year:04}}{{month:02}}/e5.oper.an.pl.128_132_v.ll025uv.{{year:04}}{{month:02}}{{day:02}}00_{{year:04}}{{month:02}}{{day:02}}23.nc"
    t_template = f"{e5_base}/e5.oper.an.pl/{{year:04}}{{month:02}}/e5.oper.an.pl.128_130_t.ll025sc.{{year:04}}{{month:02}}{{day:02}}00_{{year:04}}{{month:02}}{{day:02}}23.nc"
    z_template = f"{e5_base}/e5.oper.an.pl/{{year:04}}{{month:02}}/e5.oper.an.pl.128_129_z.ll025sc.{{year:04}}{{month:02}}{{day:02}}00_{{year:04}}{{month:02}}{{day:02}}23.nc"
    q_template = f"{e5_base}/e5.oper.an.pl/{{year:04}}{{month:02}}/e5.oper.an.pl.128_133_q.ll025sc.{{year:04}}{{month:02}}{{day:02}}00_{{year:04}}{{month:02}}{{day:02}}23.nc"
    r_template = f"{e5_base}/e5.oper.an.pl/{{year:04}}{{month:02}}/e5.oper.an.pl.128_157_r.ll025sc.{{year:04}}{{month:02}}{{day:02}}00_{{year:04}}{{month:02}}{{day:02}}23.nc"
    u10_template = f"{e5_base}/e5.oper.an.sfc/{{year:04}}{{month:02}}/e5.oper.an.sfc.128_165_10u.ll025sc.{{year:04}}{{month:02}}0100_{{year:04}}{{month:02}}{{dayend:02}}23.nc"
    v10_template = f"{e5_base}/e5.oper.an.sfc/{{year:04}}{{month:02}}/e5.oper.an.sfc.128_166_10v.ll025sc.{{year:04}}{{month:02}}0100_{{year:04}}{{month:02}}{{dayend:02}}23.nc"
    u100_template = f"{e5_base}/e5.oper.an.sfc/{{year:04}}{{month:02}}/e5.oper.an.sfc.228_246_100u.ll025sc.{{year:04}}{{month:02}}0100_{{year:04}}{{month:02}}{{dayend:02}}23.nc"
    v100_template = f"{e5_base}/e5.oper.an.sfc/{{year:04}}{{month:02}}/e5.oper.an.sfc.228_247_100v.ll025sc.{{year:04}}{{month:02}}0100_{{year:04}}{{month:02}}{{dayend:02}}23.nc"
    sp_template = f"{e5_base}/e5.oper.an.sfc/{{year:04}}{{month:02}}/e5.oper.an.sfc.128_134_sp.ll025sc.{{year:04}}{{month:02}}0100_{{year:04}}{{month:02}}{{dayend:02}}23.nc"
    msl_template = f"{e5_base}/e5.oper.an.sfc/{{year:04}}{{month:02}}/e5.oper.an.sfc.128_151_msl.ll025sc.{{year:04}}{{month:02}}0100_{{year:04}}{{month:02}}{{dayend:02}}23.nc"
    tcw_template = f"{e5_base}/e5.oper.an.sfc/{{year:04}}{{month:02}}/e5.oper.an.sfc.128_136_tcw.ll025sc.{{year:04}}{{month:02}}0100_{{year:04}}{{month:02}}{{dayend:02}}23.nc"
    t2m_template = f"{e5_base}/e5.oper.an.sfc/{{year:04}}{{month:02}}/e5.oper.an.sfc.128_167_2t.ll025sc.{{year:04}}{{month:02}}0100_{{year:04}}{{month:02}}{{dayend:02}}23.nc"

    # get the last day of the month
    dayend = calendar.monthrange(time.year, time.month)[1]

    # set the file names for the current date
    u_file = u_template.format(year=time.year, month=time.month, day=time.day)
    v_file = v_template.format(year=time.year, month=time.month, day=time.day)
    t_file = t_template.format(year=time.year, month=time.month, day=time.day)
    z_file = z_template.format(year=time.year, month=time.month, day=time.day)
    q_file = q_template.format(year=time.year, month=time.month, day=time.day)
    r_file = r_template.format(year=time.year, month=time.month, day=time.day)
    u10_file = u10_template.format(year=time.year, month=time.month, dayend=dayend)
    v10_file = v10_template.format(year=time.year, month=time.month, dayend=dayend)
    u100_file = u100_template.format(year=time.year, month=time.month, dayend=dayend)
    v100_file = v100_template.format(year=time.year, month=time.month, dayend=dayend)
    sp_file = sp_template.format(year=time.year, month=time.month, dayend=dayend)
    msl_file = msl_template.format(year=time.year, month=time.month, dayend=dayend)
    tcw_file = tcw_template.format(year=time.year, month=time.month, dayend=dayend)
    t2m_file = t2m_template.format(year=time.year, month=time.month, dayend=dayend)

    # open the files
    combined_xr = xr.open_mfdataset([u_file, v_file, t_file, z_file, q_file, r_file, u10_file, v10_file, u100_file, v100_file, sp_file, msl_file, tcw_file, t2m_file], combine='by_coords')

    # select the specific time
    combined_xr = combined_xr.sel(time=time, level = sfno_levels).squeeze()

    return combined_xr

def rda_era5_to_sfno_state(
        time : datetime.datetime,
        device : str = "cuda:0",
        e5_base : str = "/glade/campaign/collections/rda/data/ds633.0/",
        ) -> np.ndarray:
    """ Fetches the ERA5 data for a specific time and packs it into a tensor for the SFNO model.

    input:
    ------

    time : datetime.datetime
        the time to fetch the data for

    device : str
        the device to put the tensor on

    e5_base : str
        the base path to the ERA5 data

    output:
    -------

    x : torch.Tensor
        a tensor with dimensions (1, 1, 73, 721, 1440) containing the packed data
    
    """

 
    # read the data
    combined_xr = read_sfno_vars_from_era5_rda(time, e5_base = e5_base)

    # pack the data into a tensor
    x = pack_sfno_state(combined_xr, device=device)

    return x