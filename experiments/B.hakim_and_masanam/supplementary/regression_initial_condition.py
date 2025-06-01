"""
compute the regression initial conditions for the extratropical and tropical cyclone cases.

you must download ERA5 sample data to compute the regression. specifically, to repeat results in the Hakim & Masanam (2023) paper, ERA5 data are sampled every 10 days at 00UTC from 1979 to 2020.

Modified from the original by Joshua Elms
----------------------
Originator: Greg Hakim 
            ghakim@uw.edu
            University of Washington
            July 2023

"""

import numpy as np
import xarray as xr
import datetime as dt
from scipy.stats import linregress
from utils import inference
from pathlib import Path

#
# START: parameters and setup
#

# select DJF or JAS initial conditions
# ic = 'DJF'
# # set lat/lon of perturbation in degrees N, E
# ylat = 40; xlon = 150
# # localization radius in km for the scale of the initial perturbation
# locrad = 2000.
# # scaling amplitude for initial condition (1=climo variance at the base point)
# amp = -1.

ic = 'JAS'
# set lat/lon of perturbation in degrees N, E
ylat = 15.; xlon = 360.-40.
# localization radius in km for the scale of the initial perturbation
locrad = 1000.
# scaling amplitude for initial condition (1=climo variance at the base point)
amp = -1.

# netcdf data lives here
dpath = Path('/glade/derecho/scratch/jmelms/dcmip/era5')

# write regression results here:
opath = Path('/glade/derecho/scratch/jmelms/dcmip/hm24_perts')

# choose model
model = "pangu"
if model == "graphcast_small":
    raise NotImplementedError("Graphcast small model is not supported in this script, it uses a different resolution (1 degree) than these models and will need some thoughtful work before it runs.")

model_vars = {
    "sfno": {
        "param_level_pl": (
                ["z", "r", "t", "u", "v"],
                [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
            ),
        "param_sfc": ["sp", "msl", "u10", "v10", "u100", "v100", "t2m", "tcwv"],
    },
    "pangu": {
        "param_level_pl": (
                ["z", "q", "t", "u", "v"],
                [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
            ),
        "param_sfc": ["msl", "u10", "v10", "t2m"],
    },
    "graphcast_small": {
        "param_level_pl": (
                ["z", "q", "t", "u", "v", "w"],
                [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
            ),
        "param_sfc": ["msl", "u10", "v10", "t2m"],
    }
}

param_level_pl = model_vars[model]["param_level_pl"]
param, level = param_level_pl
param_sfc = model_vars[model]["param_sfc"]
nvars_pl = len(param)
nlevs = len(level)
nvars_sfc = len(param_sfc)

# names of variables used in datasets vs the names of the datasets
name_dict = {
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "u100": "100m_u_component_of_wind",
    "v100": "100m_v_component_of_wind", 
    "t2m": "2m_temperature",
    "sp": "surface_pressure",
    "msl": "mean_sea_level_pressure",
    "tcwv": "total_column_water_vapour",
    "t": "temperature",
    "z": "geopotential",
    "r": "relative_humidity",
    "q": "specific_humidity",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
}

# names used in datasets vs names used in the rest of this repo's code
name_convert_to_framework_dict = dict( 
        u10="VAR_10U",
        v10="VAR_10V",
        u100="VAR_100U",
        v100="VAR_100V",
        t2m="VAR_2T",
        sp="SP",
        msl="MSL",
        tcwv="TCW",
        t="T",
        z="Z",
        r="R",
        q="Q",
        u="U",
        v="V",
        w="W",
        lat="latitude",
        lon="longitude",
    )

#
# END: parameters and setup
#

print('computing the climo regression against one var at one point')

# ERA5 lat,lon grid
lat = 90 - np.arange(721) * 0.25
lon = np.arange(1440) * 0.25
nlat = len(lat)
nlon = len(lon)
lat_2d = np.repeat(lat[:,np.newaxis],lon.shape[0],axis=1)
lon_2d = np.repeat(lon[np.newaxis,:],lat.shape[0],axis=0)

if ic == 'DJF':
    months = ["12", "01", "02"]  # December, January, February
    pl_paths = []
    sfc_paths = []
    for m in months:
        for v in param_level_pl[0]:
            path = dpath/f'v={name_dict[v]}_m={m}.nc'
            if not path.exists():
                print(f"Warning: {path} does not exist.")
                continue
            pl_paths.append(path)
        for v in param_sfc:
            path = dpath/f'v={name_dict[v]}_m={m}.nc'
            if not path.exists():
                print(f"Warning: {path} does not exist.")
                continue
            sfc_paths.append(path)

    # set level of point to be regressed against; xlev = 5 corresponds to 500 hPa
    xlev = 5
elif ic == 'JAS':    
    months = ["07", "08", "09"]  # July, August, September
    pl_paths = []
    sfc_paths = []
    for m in months:
        for v in param_level_pl[0]:
            path = dpath/f'v={name_dict[v]}_m={m}.nc'
            if not path.exists():
                print(f"Warning: {path} does not exist.")
                continue
            pl_paths.append(path)
        for v in param_sfc:
            path = dpath/f'v={name_dict[v]}_m={m}.nc'
            if not path.exists():
                print(f"Warning: {path} does not exist.")
                continue
            sfc_paths.append(path)
    # set level of point to be regressed against; xlev = 5 corresponds to 500 hPa
    xlev = 5

else:
    raise('not a valid season. set ic to DJF or JAS')

# base point indices
bplat = int((90.-ylat)*4); bplon = int(xlon)*4
print('lat, lon=',lat[bplat],lon[bplon])

locfunc = inference.gen_circular_perturbation(lat_2d,lon_2d,bplat,bplon,1.0,locRad=locrad)
print('locfunc max:',np.max(locfunc))

# indices where this function is greater than zero
nonzeros = np.argwhere(locfunc>0.)

# indices of rectangle bounding the region (fast array access)
iminlat = np.min(nonzeros[:,0])
imaxlat = np.max(nonzeros[:,0])
iminlon = np.min(nonzeros[:,1])
imaxlon = np.max(nonzeros[:,1])
latwin = imaxlat-iminlat
lonwin = imaxlon-iminlon
print(iminlat,imaxlat,lat[iminlat],lat[imaxlat])
print(iminlon,imaxlon,lon[iminlon],lon[imaxlon])
print(latwin,lonwin)

# open pl data
pl_ds = xr.open_mfdataset(pl_paths, combine='nested', parallel=True)
# open sfc data
sfc_ds = xr.open_mfdataset(sfc_paths, combine='nested', parallel=True)

# check that the datasets have the same time dimension
assert pl_ds.sizes["valid_time"] == sfc_ds.sizes["valid_time"], "Pressure level and surface datasets must have the same number of time steps."
n_times = pl_ds.sizes["valid_time"]

# populate regression arrays
# pressure level data
regdat_pl = np.zeros([nvars_pl,n_times,nlevs,latwin,lonwin])
for i, var in enumerate(param_level_pl[0]):
    assert var in pl_ds.data_vars, f"Variable {var} not found in pressure level dataset."
    print(f"Processing variable {var} ({i+1}/{nvars_pl})")
    regdat_pl[i] = pl_ds[var].isel(latitude=slice(iminlat, imaxlat), longitude=slice(iminlon, imaxlon)).values
# surface data
regdat_sfc = np.zeros([nvars_sfc,n_times,latwin,lonwin])
for i, var in enumerate(param_sfc):
    assert var in sfc_ds.data_vars, f"Variable {var} not found in surface dataset."
    print(f"Processing variable {var} ({i+1}/{nvars_sfc})")
    regdat_sfc[i] = sfc_ds[var].isel(latitude=slice(iminlat, imaxlat), longitude=slice(iminlon, imaxlon)).values
    
# center the data
regdat_pl = regdat_pl - np.mean(regdat_pl,axis=1,keepdims=True)
regdat_sfc = regdat_sfc - np.mean(regdat_sfc,axis=1,keepdims=True)
print('\n\nregdat_pl shape:',regdat_pl.shape)
print(f"Shape comes from: {nvars_pl} pl variables x {n_times} samples x {nlevs} levels x {latwin} latitudes x {lonwin} longitudes")

for var in range(nvars_pl):
    print(var,regdat_pl[var,:,5,int(latwin/2),int(lonwin/2)])

# define the independent variable: sample at the chosen point (middle of domain)
if ic == 'DJF':
    xvar = regdat_pl[0,:,xlev,int(latwin/2)+1,int(lonwin/2)+1] # upper level
elif ic == 'JAS':
    xvar = regdat_sfc[0,:,int(latwin/2)+1,int(lonwin/2)+1] # surface

# standardize
xvar = xvar/np.std(xvar)

print('xvar shape:',xvar.shape)
print('xvar min,max:',np.min(xvar),np.max(xvar))

# regress pressure variables
regf_pl = np.zeros([nvars_pl,len(level),latwin,lonwin])
for var in range(nvars_pl):
    for k in range(len(level)):
        print('k=',k)
        for j in range(latwin):
            for i in range(lonwin):
                yvar = regdat_pl[var,:,k,j,i]
                slope,intercept,r_value,p_value,std_err = linregress(xvar,yvar)
                regf_pl[var,k,j,i] = slope*amp + intercept
                if j==latwin/2 and i == lonwin/2 and k == 5 and var == 0:
                    cov = np.matmul(xvar,yvar.T)/np.matmul(xvar,xvar.T)
                    #print('base point:',cov,slope,intercept,regf_pl[var,k,j,i])
                    
        # spatially localize
        regf_pl[var,k,:] = locfunc[iminlat:imaxlat,iminlon:imaxlon]*regf_pl[var,k,:]

# regress surface variables
regf_sfc = np.zeros([nvars_sfc,latwin,lonwin])
for var in range(nvars_sfc):
    for j in range(latwin):
        for i in range(lonwin):
            yvar = regdat_sfc[var,:,j,i]
            slope,intercept,r_value,p_value,std_err = linregress(xvar,yvar)
            regf_sfc[var,j,i] = slope*amp + intercept

    # spatially localize
    regf_sfc[var,:] = locfunc[iminlat:imaxlat,iminlon:imaxlon]*regf_sfc[var,:]

# save the regression field for later simulations
event = "cyclone" if ic == "DJF" else "hurricane"
if ylat - int(ylat) == 0:
    str_lat = f"{int(ylat)}"
else:
    str_lat = f"{round(ylat*4)/4:.2f}"
if xlon - int(xlon) == 0:
    str_lon = f"{int(xlon)}"
else:
    str_lon = f"{round(xlon*4)/4:.2f}"
rgfile = opath / f'{event}_{ic}_{str_lat}N_{str_lon}E_regression_{model}.nc'

ds = xr.Dataset(
    coords={
        "level": level,
        "latitude": lat[iminlat:imaxlat],
        "longitude": lon[iminlon:imaxlon],
    },
)

for i, var in enumerate(param_level_pl[0]):
    ds[name_convert_to_framework_dict[var]] = xr.DataArray(
        regf_pl[i], dims=["level", "latitude", "longitude"], attrs={"long_name": name_dict[var]}
    )
for i, var in enumerate(param_sfc):
    ds[name_convert_to_framework_dict[var]] = xr.DataArray(
        regf_sfc[i], dims=["latitude", "longitude"], attrs={"long_name": name_dict[var]}
    )
if rgfile.exists():
    print(f"Warning: {rgfile} already exists. Overwriting.")
    rgfile.unlink()  # remove the existing file
ds.to_netcdf(rgfile, mode='w', format='NETCDF4')

print(f"Regression fields saved to {rgfile}")