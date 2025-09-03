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
import os
from scipy.stats import linregress
from utils_E2S import general
from pathlib import Path
from download_and_compute_ICs import generate_dates
import multiprocessing as mp
from time import perf_counter

MODEL_LEVELS = dict(
    SFNO=[1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    Pangu6=[1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    GraphCastOperational=[
        1000,
        925,
        850,
        700,
        600,
        500,
        400,
        300,
        250,
        200,
        150,
        100,
        50,
    ],
    FuXi=[1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
)


def linreg(xvar, regdat, regf, var, j, i):
    slope, intercept, r_value, p_value, std_err = linregress(xvar, regdat[var, :, j, i])
    regf[var, j, i] = slope * amp + intercept


def compute_regression(
    year_range: list[int],
    ic_months: list[int],
    ic_str: str,
    ylat: float,
    xlon: float,
    locrad: float,
    amp: float,
    rpath: Path,
    dpath: Path,
    opath: Path,
    model: str,
    plev: str | int = 500,
):
    """
    Compute the regression initial conditions for the specified model and season.
    Parameters:
        year_range (list[int]): List of years to include in the regression, e.g. [2018, 2019].
        ic_months (list[int]): List of months to use for initial conditions (1-12), e.g. [12, 1, 2] for DJF.
        ic_str (str): Initial condition season, either 'DJF' for NH winter or 'JAS' for NH summer.
        ylat (float): Latitude of the perturbation in degrees North.
        xlon (float): Longitude of the perturbation in degrees East.
        locrad (float): Localization radius in kilometers for the scale of the initial perturbation.
        amp (float): Scaling amplitude for the initial condition (1 = climo variance at the base point).
        rpath (Path): Path to the raw data.
        dpath (Path): Path to the model initial conditions.
        opath (Path): Path to the directory where the regression results will be saved.
        model (str): The model to use for regression, from: ["SFNO", "Pangu6", "GraphCastOperational", "FuXi"].
        plev (int): Pressure level to regress against (default: 500 hPa). To use surface pressure ('msl' if present, 'sp' if not), pass 'sfc'.
    """
    SL_VARIABLES = ["msl", "sp", "t2m", "tp06", "tcwv", "u10", "v10", "u100", "v100"]
    PL_VARIABLES = ["z", "r", "t", "u", "v", "q", "w"]
    INVARIANT_VARIABLES = ["z", "lsm"]

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
        "tp06": "total_precipitation_06",
        "lsm": "land_sea_mask",
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
        u10="u10m",
        v10="v10m",
        u100="u100m",
        v100="v100m",
        tp="tp06",
    )

    input_data_path = dpath / f"{model}.nc"
    if not input_data_path.exists():
        raise FileNotFoundError(f"Input data file {input_data_path} does not exist.")

    ds = xr.open_dataset(input_data_path)
    nvars = len(ds.data_vars)
    levs = MODEL_LEVELS[model]
    nlevs = len(levs)
    # figure out which variables on which levels are in this dataset
    params_pl, params_sl, params_invariant = [], [], []
    for var in ds.data_vars:
        if var in SL_VARIABLES:
            params_sl.append(var)
        elif (
            len(var) > 1 and var[0] in PL_VARIABLES
        ):  # need to keep out invariant "z" which matches PL name
            try:
                var_lev = int(var[1:])
                if var_lev in levs:
                    params_pl.append(var)
            except Exception:
                pass
        elif var in INVARIANT_VARIABLES:
            params_invariant.append(var)
        else:
            raise ValueError(f"Unknown variable {var} found in dataset.")

    if "msl" not in params_sl and "sp" not in params_sl and plev == "sfc":
        raise ValueError(
            "At least one of 'msl' or 'sp' must be present in surface level variables to use plev='sfc' ."
        )
    nvars_pl = len(params_pl)
    nvars_sl = len(params_sl)
    nvars_invariant = len(params_invariant)
    relevant_vars = (
        params_pl + params_sl
    )  # we don't want to work with invariant vars, the pert for those will be 0 and added at end
    nvars_rel = len(relevant_vars)
    try:
        z_str = f"z{plev}"
        xlev = relevant_vars.index(z_str)
        print(f"Using {z_str} as independent variable for regression.")
    except ValueError:
        if plev != "sfc":
            raise ValueError(
                f"Geopotential field {z_str} not found in model levels {levs}."
            )
        else:
            if "msl" in relevant_vars:
                xlev = relevant_vars.index("msl")
            else:
                xlev = relevant_vars.index("sp")

    # figure out which files need to be opened for this data
    dates = generate_dates(year_range, ic_months)[:2]
    n_times = len(dates)
    raw_paths = []

    for date in dates:
        for vp in params_pl:
            v, p = vp[0], vp[1:]
            path = rpath / f"v={name_dict[v]}_p={p}_d={date}.nc"
            if not path.exists():
                print(f"file no existe: {path}")
            raw_paths.append(path)
        for v in params_sl:
            path = rpath / f"v={name_dict[v]}_d={date}.nc"
            if not path.exists():
                print(f"file no existe: {path}")
            raw_paths.append(path)

    print("computing the climo regression against one var at one point")

    # ERA5 lat,lon grid
    lat = 90 - np.arange(721) * 0.25
    lon = np.arange(1440) * 0.25
    nlat = len(lat)
    nlon = len(lon)
    lat_2d = np.repeat(lat[:, np.newaxis], lon.shape[0], axis=1)
    lon_2d = np.repeat(lon[np.newaxis, :], lat.shape[0], axis=0)

    # base point indices
    bplat = int((90.0 - ylat) * 4)
    bplon = int(xlon) * 4
    print("lat, lon=", lat[bplat], lon[bplon])

    locfunc = general.gen_circular_perturbation(
        lat_2d, lon_2d, bplat, bplon, 1.0, locRad=locrad
    )
    print("locfunc max:", np.max(locfunc))

    # indices where this function is greater than zero
    nonzeros = np.argwhere(locfunc > 0.0)

    # indices of rectangle bounding the region (fast array access)
    iminlat = np.min(nonzeros[:, 0])
    imaxlat = np.max(nonzeros[:, 0])
    iminlon = np.min(nonzeros[:, 1])
    imaxlon = np.max(nonzeros[:, 1])
    latwin = imaxlat - iminlat
    lonwin = imaxlon - iminlon
    print(iminlat, imaxlat, lat[iminlat], lat[imaxlat])
    print(iminlon, imaxlon, lon[iminlon], lon[imaxlon])
    print(latwin, lonwin)

    # open all data files
    print(f"Opening {len(raw_paths)} raw data files... ")
    start = perf_counter()
    raw_ds = xr.open_mfdataset(raw_paths, combine="nested", parallel=True)
    stop = perf_counter()
    print(
        f"Opened {len(raw_paths)} files in {stop - start:.2f} s, avg'ing {(stop - start) / (1000*len(raw_paths)):.2f} ms/file."
    )
    # populate regression arrays
    regdat = np.zeros([nvars_rel, n_times, latwin, lonwin])
    for i, var in enumerate(relevant_vars):
        print(f"Processing variable {var} ({i+1}/{nvars_rel})")
        if i < nvars_pl:  # pressure levels
            v, p = var[0], var[1:]
            regdat[i] = (
                raw_ds[v]
                .isel(
                    latitude=slice(iminlat, imaxlat), longitude=slice(iminlon, imaxlon)
                )
                .sel(pressure_level=p)
                .values
            )
        else:  # single levels
            regdat[i] = (
                raw_ds[var]
                .isel(
                    latitude=slice(iminlat, imaxlat), longitude=slice(iminlon, imaxlon)
                )
                .values
            )

    # center the data
    regdat = regdat - np.mean(regdat, axis=1, keepdims=True)
    print("\n\nregdat shape:", regdat.shape)
    print(
        f"Shape comes from: {nvars_rel} variables x {n_times} samples x {latwin} latitudes x {lonwin} longitudes"
    )

    # for var in range(nvars_rel):
    #     print(var,regdat[var,:,5,int(latwin/2),int(lonwin/2)])

    # define the independent variable: sample at the chosen point (middle of domain)
    xvar = regdat[xlev, :, int(latwin / 2) + 1, int(lonwin / 2) + 1]

    # standardize
    xvar = xvar / np.std(xvar)

    print("xvar shape:", xvar.shape)
    print("xvar min,max:", np.min(xvar), np.max(xvar))

    # set up arg list to pass to mp.Pool
    # this was a slow loop, so it's parallel now
    # regress variables
    start = perf_counter()
    args = []
    regf = np.zeros([nvars_rel, latwin, lonwin])
    for var in range(nvars_rel):
        for j in range(latwin):
            for i in range(lonwin):
                args.append((xvar, regdat, regf, var, j, i))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(linreg, args)

    for var in range(nvars_rel):
        # spatially localize
        regf[var, :] = locfunc[iminlat:imaxlat, iminlon:imaxlon] * regf[var, :]

    stop = perf_counter()
    print(
        f"Regression computation completed in {stop - start:.2f} seconds, avg'ing {(stop - start) / (n_times):.2f} s/date."
    )

    # save the regression field for later simulations
    if ylat - int(ylat) == 0:
        str_lat = f"{int(ylat)}"
    else:
        str_lat = f"{round(ylat*4)/4:.2f}"
    if xlon - int(xlon) == 0:
        str_lon = f"{int(xlon)}"
    else:
        str_lon = f"{round(xlon*4)/4:.2f}"
    rgfile = opath / f"{ic_str}_{str_lat}N_{str_lon}E_z-regression_{model}.nc"

    output_ds = xr.Dataset(
        coords={
            "latitude": lat,
            "longitude": lon,
        },
    )

    zero_da = xr.DataArray(np.zeros((nlat, nlon)), dims=["latitude", "longitude"])

    for i, var in enumerate(relevant_vars):
        if i == 0:
            breakpoint()
        da_name = name_convert_to_framework_dict.get(var, var)
        output_ds[da_name] = zero_da.copy()
        output_ds[da_name][
            dict(latitude=slice(iminlat, imaxlat), longitude=slice(iminlon, imaxlon))
        ] = regf[i]
    for i, var in enumerate(params_invariant):
        da_name = name_convert_to_framework_dict.get(var, var)
        output_ds[da_name] = zero_da.copy()

    if rgfile.exists():
        print(f"Warning: {rgfile} already exists. Overwriting.")
        rgfile.unlink()  # remove the existing file
    output_ds.to_netcdf(rgfile, mode="w", format="NETCDF4")

    print(f"Regression fields saved to {rgfile}")


#
# START: parameters and call to compute_regression
#

# raw data
rpath = Path("/N/slate/jmelms/projects/HM24_ICs/raw")

# netcdf full initial conditions lives here
dpath = Path("/N/slate/jmelms/projects/HM24_ICs/DJF_2018-2019/IC_files")

# write regression results here:
opath = Path("/N/slate/jmelms/projects/HM24_ICs/DJF_2018-2019/reg_pert_files")
opath.mkdir(parents=True, exist_ok=True)

# set dates
start_end_years_inc = [2018, 2019]
year_range = range(
    start_end_years_inc[0], start_end_years_inc[1] + 1
)  # inclusive range

# select DJF or JAS initial conditions
ic_months = [12, 1, 2]
ic_str = "DJF"
# set lat/lon of perturbation in degrees N, E
ylat = 40
xlon = 150
# localization radius in km for the scale of the initial perturbation
locrad = 2000.0
# scaling amplitude for initial condition (1=climo variance at the base point)
amp = -1.0

models = ["SFNO", "Pangu6", "GraphCastOperational", "FuXi"]
for model in models:
    compute_regression(
        year_range,
        ic_months,
        ic_str,
        ylat,
        xlon,
        locrad,
        amp,
        rpath,
        dpath,
        opath,
        model=model,
    )


ic_months = [7, 8, 9]
ic_str = "JAS"
# set lat/lon of perturbation in degrees N, E
ylat = 15.0
xlon = 360.0 - 40.0
# localization radius in km for the scale of the initial perturbation
locrad = 1000.0
# scaling amplitude for initial condition (1=climo variance at the base point)
amp = -1.0

for model in models:
    compute_regression(
        year_range,
        ic_months,
        ic_str,
        ylat,
        xlon,
        locrad,
        amp,
        rpath,
        dpath,
        opath,
        model=model,
    )

#
# END: parameters and call to compute_regression
#
