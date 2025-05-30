import datetime as dt
import calendar
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
sfno_2dvars = ["VAR_10U", "VAR_10V", "VAR_100U", "VAR_100V", "VAR_2T", "SP", "MSL", "TCW"]
nlat = 721
nlon = 1440

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
        - T2M : (time, latitude, longitude) float32 >MOD< betting copilot snuck this one in

        Note that the following specific levels are expected to be available: 
        [50,100,150,200,250,300,400,500,600,700,850,925,1000] 

    device : str
        the device to put the tensor on

    output:
    -------

    x : torch.Tensor
        a tensor with dimensions (1, 1, 73, 721, 1440) containing the packed data (ensemble = 1, time = 1, channels = 73, lat = 721, lon = 1440)

    The data are packed as follows:

    channel = "u10m", "v10m", "u100m", "v100m", "t2m", "sp", "msl", "tcwv", 
        "u50", "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", 
        "u700", "u850", "u925", "u1000", "v50", "v100", "v150", "v200", "v250", 
        "v300", "v400", "v500", "v600", "v700", "v850", "v925", "v1000", "z50", 
        "z100", "z150", "z200", "z250", "z300", "z400", "z500", "z600", "z700", 
        "z850", "z925", "z1000", "t50", "t100", "t150", "t200", "t250", "t300", 
        "t400", "t500", "t600", "t700", "t850", "t925", "t1000", "r50", "r100", 
        "r150", "r200", "r250", "r300", "r400", "r500", "r600", "r700", "r850", 
        "r925", "r1000" ;

    """
    # latitudes should be decreasing for SFNO
    ds = ds.sortby('latitude', ascending=False)

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
        x2d = ds["VAR_10U"].squeeze()
        x2d = xr.concat((x2d, ds["VAR_10V"].squeeze()), dim = "n")
        x2d = xr.concat((x2d, ds["VAR_100U"].squeeze()), dim = "n")
        x2d = xr.concat((x2d, ds["VAR_100V"].squeeze()), dim = "n")
        x2d = xr.concat((x2d, ds["VAR_2T"].squeeze()), dim = "n")
        x2d = xr.concat((x2d, ds["SP"].squeeze()), dim = "n")
        x2d = xr.concat((x2d, ds["MSL"].squeeze()), dim = "n")
        x2d = xr.concat((x2d, ds["TCW"].squeeze()), dim = "n")

        # concatenate the 2d and 3d variables
        x = xr.concat((x2d, x3d), dim = "n")

    # convert to a torch array of type float32
    x = torch.from_numpy(x.values).to(device=device).float()

    # add ensemble and time dimension
    if len(x.shape) == 4:
        x = x[None, ...]
    elif len(x.shape) == 3:
        x = x[None, None, ...]

    return x

def read_sfno_vars_from_era5_rda(
        time : dt.datetime,
        e5_base : str = "/glade/campaign/collections/rda/data/ds633.0/",
        ) -> xr.Dataset:
    """ Fetches the ERA5 data for a specific time necessary for running the SFNO model. 
    
    input:
    ------

    time : dt.datetime
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
        time : dt.datetime,
        device : str = "cpu",
        e5_base : str = "/glade/campaign/collections/rda/data/ds633.0/",
        ) -> torch.tensor:
    """ Fetches the ERA5 data for a specific time and packs it into a tensor for the SFNO model.

    input:
    ------

    time : dt.datetime
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

def initialize_sfno_xarray_ds(time, n_ensemble : int) -> xr.Dataset:
    """ Initializes an xarray dataset for output from the SFNO model. 
    
    input:
    -----

    time : 
        the time(s) to assign to the data

    n_ensemble : int
        the number of ensemble members
    
    output:
    -------
    ds : xr.Dataset
        the initialized dataset
    """

    # create the dataset
    ds = xr.Dataset()

    # check if we need to put time into an array/list
    if not isinstance(time, (list, tuple, np.ndarray)):
        time = [time]

    # explicitly cast to datetime64[ns] to avoid warning
    time = np.array(time, dtype='datetime64[ns]')
    
    # create the coordinates
    ds['time'] = xr.DataArray(time, dims='time')
    ds['level'] = xr.DataArray(sfno_levels, dims='level')
    ds['latitude'] = xr.DataArray(np.linspace(-90,90,nlat), dims='latitude')
    ds['longitude'] = xr.DataArray(np.linspace(0,359.75,nlon), dims='longitude')
    ds['ensemble'] = xr.DataArray(np.arange(n_ensemble), dims='ensemble')

    # add metadata to the coordinates
    ds['time'].attrs['long_name'] = 'time'
    ds['level'].attrs['long_name'] = 'pressure level'
    ds['latitude'].attrs['long_name'] = 'latitude'
    ds['longitude'].attrs['long_name'] = 'longitude'
    ds['ensemble'].attrs['long_name'] = 'ensemble member'
    ds['level'].attrs['units'] = 'hPa'
    ds['latitude'].attrs['units'] = 'degrees_north'
    ds['longitude'].attrs['units'] = 'degrees_east'

    return ds

def set_sfno_xarray_metadata(ds : xr.Dataset, copy : bool = False) -> xr.Dataset:
    """Sets the metadata for the SFNO model xarray dataset.
    
    input:
    ------
    ds : xr.Dataset
        the dataset to set the metadata for

    copy : bool
        if True, the dataset is copied before setting the metadata

    output:
    -------
    ds : xr.Dataset
        the dataset with the metadata set
    """
    if copy:
        # copy the dataset
        ds = ds.copy()

    # set metadata
    all_vars = sfno_2dvars + sfno_3dvars
    md = { v : {} for v in all_vars }
    md['U']['long_name'] = 'zonal wind'
    md['U']['units'] = 'm/s'
    md['V']['long_name'] = 'meridional wind'
    md['V']['units'] = 'm/s'
    md['Z']['long_name'] = 'geopotential height'
    md['Z']['units'] = 'm**2 s**-2'
    md['T']['long_name'] = 'temperature'
    md['T']['units'] = 'K'
    md['R']['long_name'] = 'relative humidity'
    md['R']['units'] = '%'
    md['VAR_10U']['long_name'] = '10m zonal wind'
    md['VAR_10U']['units'] = 'm/s'
    md['VAR_10V']['long_name'] = '10m meridional wind'
    md['VAR_10V']['units'] = 'm/s'
    md['VAR_100U']['long_name'] = '100m zonal wind'
    md['VAR_100U']['units'] = 'm/s'
    md['VAR_100V']['long_name'] = '100m meridional wind'
    md['VAR_100V']['units'] = 'm/s'
    md['VAR_2T']['long_name'] = '2m temperature'
    md['VAR_2T']['units'] = 'K'
    md['SP']['long_name'] = 'surface pressure'
    md['SP']['units'] = 'Pa'
    md['MSL']['long_name'] = 'mean sea level pressure'
    md['MSL']['units'] = 'Pa'
    md['TCW']['long_name'] = 'total column water'
    md['TCW']['units'] = 'kg/m^2'

    # loop over the variables and set the metadata
    for var in all_vars:
        for attr, val in md[var].items():
            ds[var].attrs[attr] = val

    return ds

def unpack_sfno_state(
        x : torch.tensor,
        time = None,
        ) -> xr.Dataset:
    """ Unpacks the SFNO model state tensor into an xarray dataset.

    input:
    ------

    x : torch.Tensor

    time :
        the time(s) to assign to the data

    
    output:
    -------

    ds : xr.Dataset

    """

    # get the number of ensemble members
    n_ensemble = x.shape[1]

    # deal with default time
    if time is None:
        n_time = x.shape[0]
        time = [dt.datetime(1850,1,1) + dt.timedelta(hours=i*6) for i in range(n_time)]

    # initialize the dataset
    ds = initialize_sfno_xarray_ds(time, n_ensemble)

    # loop over the 2D variables and insert them into the dataset
    for i, var in enumerate(sfno_2dvars):
        ds[var] = xr.DataArray(x[:,:,i,:,:].flip(-2).cpu().numpy(), dims=('time', 'ensemble', 'latitude', 'longitude'))

    n2d = len(sfno_2dvars)
    nlev = len(sfno_levels)
    # loop over the 3D variables and insert them into the dataset
    for j, var in enumerate(sfno_3dvars):
        # get the indices for the current 3D variable
        i1 = n2d + j*nlev
        i2 = n2d + (j+1)*nlev

        # load the 3D variables
        ds[var] = xr.DataArray(x[:,:,i1:i2,:,:].flip(-2).cpu().numpy(), dims=('time', 'ensemble', 'level', 'latitude', 'longitude'))

    return ds

def create_empty_sfno_ds(
        n_ensemble : int = 1,
        time : dt.datetime = dt.datetime(1850,1,1),
        ) -> xr.Dataset:
    """ Initializes an empty xarray dataset for the SFNO model.

    input:
    ------

    n_ensemble : int
        the number of ensemble members

    time : dt.datetime
        the time to assign to the data

    output:
    -------

    ds : xr.Dataset
        the initialized dataset
    
    """

    # initialize the dataset
    ds = initialize_sfno_xarray_ds(time, n_ensemble)

    # loop over the 2D variables and insert empty arrays
    for var in sfno_2dvars:
        ds[var] = xr.DataArray(np.empty((1, n_ensemble, nlat, nlon)), dims=('time', 'ensemble', 'latitude', 'longitude'))

    # loop over the 3D variables and insert empty arrays
    for var in sfno_3dvars:
        ds[var] = xr.DataArray(np.empty((1, n_ensemble, len(sfno_levels), nlat, nlon)), dims=('time', 'ensemble', 'level', 'latitude', 'longitude'))

    # set the metadata
    ds = set_sfno_xarray_metadata(ds)

    return ds

def slow_latitude_weighted_mean(da, latitudes):
    """
    [Deprecated] - Use `latitude_weighted_mean` instead.
    Calculate the latitude weighted mean of a variable in a dataset
    """
    lat_radians = np.deg2rad(latitudes)
    weights = np.cos(lat_radians)
    weights.name = "weights"
    var_weighted = da.weighted(weights)
    return var_weighted.mean(dim=["latitude", "longitude"])

def latitude_weighted_mean(da, latitudes, device="cpu"):
    """
    Calculate the latitude weighted mean of a variable using torch operations on GPU.
    Needs tests to ensure it works correctly.
    
    Parameters:
    -----------
    da : xarray.DataArray or torch.Tensor
        The data to average
    latitudes : xarray.DataArray or numpy.ndarray
        The latitude values
        
    Returns:
    --------
    torch.Tensor
        The latitude-weighted mean
    """
    # Convert inputs to torch tensors if needed
    coords = {dim: da[dim] for dim in da.dims if dim not in ['latitude', 'longitude']}
    if isinstance(da, xr.DataArray):
        da = torch.from_numpy(da.values)
    if isinstance(latitudes, xr.DataArray):
        latitudes = latitudes.values
    
    # Move to GPU if available
    da = da.to(device)
    
    # Calculate weights
    lat_radians = torch.from_numpy(np.deg2rad(latitudes)).to(device)
    weights = torch.cos(lat_radians) / (torch.sum(torch.cos(lat_radians)) * da.shape[-1])
    
    # Expand weights to match data dimensions
    weights = weights.view(1, -1, 1)  # Add dims for broadcasting
    
    # Calculate weighted mean
    weighted_data = da * weights
    averaged = weighted_data.nansum(dim=(-2, -1))  # Average over lat, lon dimensions
    return xr.DataArray(averaged.cpu().numpy(), coords=coords)

def single_IC_inference(
        model,
        n_timesteps : int,
        init_time : dt.datetime = dt.datetime(1850,1,1),
        initial_condition : xr.Dataset = None,
        initial_perturbation : xr.Dataset = None,
        recurrent_perturbation : xr.Dataset = None,
        device : str = "cpu",
        vocal: bool = False,
        ) -> xr.Dataset:
    """ Runs the SFNO model for a single initial condition.
    input:
    ------
    model : torch.nn.Module
        the SFNO model to run
        
    n_timesteps : int
        the number of timesteps to run the model for

    init_time : dt.datetime
        the time to initialize the model at. if providing initial_condition, this doesn't need to be set

    initial_condition (optional) : xr.Dataset
        the initial condition to use for the model. If not provided, the model will be initialized using the rda_era5_to_sfno_state function at init_time.

    perturbation (optional) : xr.Dataset
        the perturbation to apply to the initial condition. This is added to the initial condition before running the model.
        If not provided, the model will be run with the initial condition only.
        Must be of the same shape as the initial condition.

    device : str
        the device to run the model on
        
    vocal : bool
        if True, print progress messages
        
    output:
    -------

    ds_out : xr.Dataset
        the output dataset from the model
    
    """
    timedeltas = np.arange(0, 6*(n_timesteps+1), 6) * np.timedelta64(1, 'h')
    end_time = init_time + dt.timedelta(hours=6*(n_timesteps))
    
    if initial_condition is not None:
        # pack the initial condition into a tensor
        x = pack_sfno_state(initial_condition, device=device)
        
    else:
        # just use rda_era5_to_sfno_state to get the initial condition
        x = rda_era5_to_sfno_state(device=device, time = init_time)
        
    # check if we need to apply a perturbation
    if initial_perturbation is not None:
        # pack the perturbation into a tensor
        xpert = pack_sfno_state(initial_perturbation, device=device)
        
        # add the perturbation to the initial condition
        x = x + xpert

    # handle perturbation which will be added to the model output at every timestep
    if recurrent_perturbation is not None:
        rpert = pack_sfno_state(recurrent_perturbation, device=device)
        
    else:
        rpert = None
        
    # run the model
    data_list = [] # output accumulator
    iterator = model(init_time, x, rpert)
    for k, (time, data, _) in enumerate(iterator):
        if vocal:
            print(f"Step {k}: {time} completed.")

        # append the data to the list
        # (move the data to the cpu (memory))
        data_list.append(data.cpu())

        # check if we're at the end time
        if time >= end_time:
            break

    # stack the output data by time
    data = torch.stack(data_list)
    
    return unpack_sfno_state(data, time = timedeltas)

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    
    Adapted from Hakim and Masanam (2024) repository: https://github.com/modons/DL-weather-dynamics/blob/main/panguweather_utils.py#L145
    """

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = list(map(np.radians, [lon1, lat1, lon2, lat2]))
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367.0 * c
    return km

def gen_circular_perturbation(lat_2d,lon_2d,ilat,ilon,amp,locRad=1000.,Z500=False):
    """
    Adapted from Hakim and Masanam (2024) repository: https://github.com/modons/DL-weather-dynamics/blob/main/panguweather_utils.py#L162
    """
    grav = 9.81
    nlat = lat_2d.shape[0]
    nlon = lon_2d.shape[1]
    site_lat = lat_2d[ilat,0]
    site_lon = lon_2d[0,ilon]
    lat_vec = np.reshape(lat_2d,[nlat*nlon])
    lon_vec = np.reshape(lon_2d,[nlat*nlon])
    dists = np.zeros(shape=[nlat*nlon])
    dists = np.array(haversine(site_lon,site_lat,lon_vec,lat_vec),dtype=np.float64)

    hlr = 0.5*locRad # work with half the localization radius
    r = dists/hlr

    # indexing w.r.t. distances
    ind_inner = np.where(dists <= hlr)    # closest
    ind_outer = np.where(dists >  hlr)    # close
    ind_out   = np.where(dists >  2.*hlr) # out

    # Gaspari-Cohn function
    covLoc = np.ones(shape=[nlat*nlon],dtype=np.float64)

    # for pts within 1/2 of localization radius
    covLoc[ind_inner] = (((-0.25*r[ind_inner]+0.5)*r[ind_inner]+0.625)* \
                         r[ind_inner]-(5.0/3.0))*(r[ind_inner]**2)+1.0
    # for pts between 1/2 and one localization radius
    covLoc[ind_outer] = ((((r[ind_outer]/12. - 0.5) * r[ind_outer] + 0.625) * \
                          r[ind_outer] + 5.0/3.0) * r[ind_outer] - 5.0) * \
                          r[ind_outer] + 4.0 - 2.0/(3.0*r[ind_outer])
    # Impose zero for pts outside of localization radius
    covLoc[ind_out] = 0.0
    
    # prevent negative values: calc. above may produce tiny negative
    covLoc[covLoc < 0.0] = 0.0
    
    if Z500:
        # 500Z:
        print('500Z perturbation...')
        perturb = np.reshape(covLoc*grav*amp,[nlat,nlon])
    else:
        # heating:
        print('heating perturbation...')
        perturb = np.reshape(covLoc*amp,[nlat,nlon])

    return perturb

def gen_elliptical_perturbation(lat,lon,k,ylat,xlon,locRad):

    """
    center a localized ellipse at (xlat,xlon)
    
    Adapted from Hakim and Masanam (2024) repository: https://github.com/modons/DL-weather-dynamics/blob/main/panguweather_utils.py#L208
    
    k: meridional wavenumber; disturbance is non-zero up to first zero crossing in cos
    xlat: latitude, in degrees to center the function
    xlon: longitude, in degrees to center the function
    locRad: zonal GC distance, in km
    """
    km = 1.e3
    nlat = len(lat)
    nlon = len(lon)
 
    ilon = xlon*4. #lon index of center
    ilat = int((90.-ylat)*4.) #lat index of center
    yfunc = np.cos(np.deg2rad(k*(lat-ylat)))

    # first zero-crossing
    crit = np.cos(np.deg2rad(k*(lat[ilat]-ylat)))
    ll = np.copy(ilat)
    while crit>0:
        ll-=1
        crit = yfunc[ll]

    yfunc[:ll+1] = 0.
    yfunc[2*ilat-ll:] = 0.

    # gaspari-cohn in logitude only, at the equator
    dx = 6380.*km*2*np.pi/(360.) #1 degree longitude at the equator
    dists = np.zeros_like(lon)
    for k in range(len(lon)):
        dists[k] = dx*np.min([np.abs(lon[k]-xlon),np.abs(lon[k]-360.-xlon)])

    #locRad = 10000.*km
    hlr = 0.5*locRad # work with half the localization radius
    r = dists/hlr

    # indexing w.r.t. distances
    ind_inner = np.where(dists <= hlr)    # closest
    ind_outer = np.where(dists >  hlr)    # close
    ind_out   = np.where(dists >  2.*hlr) # out

    # Gaspari-Cohn function
    covLoc = np.ones(nlon)

    # for pts within 1/2 of localization radius
    covLoc[ind_inner] = (((-0.25*r[ind_inner]+0.5)*r[ind_inner]+0.625)* \
                         r[ind_inner]-(5.0/3.0))*(r[ind_inner]**2)+1.0
    # for pts between 1/2 and one localization radius
    covLoc[ind_outer] = ((((r[ind_outer]/12. - 0.5) * r[ind_outer] + 0.625) * \
                          r[ind_outer] + 5.0/3.0) * r[ind_outer] - 5.0) * \
                          r[ind_outer] + 4.0 - 2.0/(3.0*r[ind_outer])
    # Impose zero for pts outside of localization radius
    covLoc[ind_out] = 0.0

    # prevent negative values: calc. above may produce tiny negative
    covLoc[covLoc < 0.0] = 0.0

    # make the function
    [a,b] = np.meshgrid(covLoc,yfunc)
    perturb = a*b

    return perturb

def gen_baroclinic_wave_perturbation(lat,lon,ylat,xlon,u_pert_base,locRad,a=6.371e6):
    """
    Implementation of baroclinic wave perturbation from Bouvier et al. (2024). 
    
    Produces a "localised unbalanced [u-]wind perturbation" to be added to a "baroclinically unstable background state".
    
    input:
    ------
    lat : numpy.ndarray
        the latitude values
    lon : numpy.ndarray
        the longitude values
    ylat : float
        the latitude of the center of the perturbation
    xlon : float
        the longitude of the center of the perturbation
    u_pert_base : float
        the base amplitude of the u-wind perturbation
    locRad : float
        the localization radius (approximate size of perturbation) in km 
    a (optional) : float
        the radius of the earth in m (default is 6.371e6 m)
        
    output:
    -------
    perturb : numpy.ndarray
        the perturbation array with shape (nlat, nlon)
    """
    radlat = np.deg2rad(lat)
    radlon = np.deg2rad(lon)
    radylat = np.deg2rad(ylat)
    radxlon = np.deg2rad(xlon)
    
    # make the grid
    lon_2d, lat_2d = np.meshgrid(radlon, radlat)

    # calculate distance from center of perturbation for each grid point
    great_circle_dist = a*np.arccos(
        np.sin(radylat) * np.sin(lat_2d) + 
        np.cos(radylat) * np.cos(lat_2d) * np.cos(lon_2d - radxlon)
    )
    perturb = u_pert_base * np.exp(-(great_circle_dist / locRad)**2)
    
    return perturb

if __name__=="__main__": 
    # test the functions
    lat = np.arange(90, -90.001, -0.25)
    lon = np.arange(0, 360, 0.25)
    ylat = 0. # deg N
    xlon = 20. # deg E
    a = 6.371e6 # radius of earth in m
    locRad = a/10 # 10% of the radius of the earth
    u_pert_base = 1.0 # m/s
    pert = gen_baroclinic_wave_perturbation(
        lat, lon, ylat, xlon, u_pert_base, locRad
    )
    import matplotlib.pyplot as plt
    plt.imshow(pert)
    plt.colorbar()
    plt.savefig("pert.png")
    plt.close()