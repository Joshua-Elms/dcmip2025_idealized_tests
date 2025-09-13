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

from utils import general, model_info
import download_and_compute_ICs as IC
import numpy as np
import xarray as xr
from pathlib import Path
from scipy.stats import linregress
from time import perf_counter


def compute_regression(
    year_range: list[int],
    ic_months: list[int],
    ic_str: str,
    ylat: float,
    xlon: float,
    locrad: float,
    amp: float,
    rpath: Path,
    opath: Path,
    model: str,
    independent_var: str = "z500",
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
        opath (Path): Path to the directory where the regression results will be saved.
        model (str): The model to use for regression, from: ["super", "SFNO", "Pangu6", "GraphCastOperational", "FuXi"]. "super" is all variables.
        independent_var (str): Variable to regress against, e.g. 'z500' or 'msl'.
    """
    if model == "super":
        var_names = model_info.MASTER_VARIABLES_NAMES
        var_types = model_info.MASTER_VARIABLES_TYPES
    else:
        var_names = model_info.MODEL_VARIABLES.get(model)["names"]
        var_types = model_info.MODEL_VARIABLES.get(model)["types"]
        
    nvars = len(var_names)

    # figure out which variables are in this model
    params_pl, params_sl, params_invariant = [], [], []

    for i in range(nvars):
        if var_types[i] == model_info.PL:
            params_pl.append(var_names[i])
        elif var_types[i] == model_info.SL:
            params_sl.append(var_names[i])
        elif var_types[i] == model_info.IN:
            params_invariant.append(var_names[i])
        else:
            raise ValueError(f"Unknown variable type for {var_names[i]}.")

    nvars_pl = len(params_pl)
    nvars_sl = len(params_sl)
    nvars_invariant = len(params_invariant)
    relevant_vars = (
        params_pl + params_sl
    )  # we don't want to work with invariant vars, the pert for those will be 0 and added at end
    nvars_rel = len(relevant_vars)
    try:
        idx_var = relevant_vars.index(independent_var)
        print(f"Using {independent_var} as independent variable for regression.")
    except ValueError:
        raise ValueError(
            f"Independent variable {independent_var} not found in model {model} variables."
        )

    # figure out which files need to be opened for this data
    dates = IC.generate_dates(year_range, ic_months)
    n_times = len(dates)
    print(f"n_times: {n_times}, dates: {dates}")
    raw_paths_by_var = {var: [] for var in relevant_vars}

    for date in dates:
        for vp in params_pl:
            v, p = vp[0], vp[1:]
            path = rpath / f"v={model_info.E2S_TO_CDS[v]}_l={p}_d={date}.nc"
            if not path.exists():
                print(f"file no existe: {path}")
            raw_paths_by_var[vp].append(path)
        for v in params_sl:
            path = rpath / f"v={model_info.E2S_TO_CDS[v]}_l=sl_d={date}.nc"
            if not path.exists():
                print(f"file no existe: {path}")
            raw_paths_by_var[v].append(path)

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

    # populate regression arrays
    regdat = np.zeros([nvars_rel, n_times, latwin, lonwin])
    for i, var in enumerate(relevant_vars):
        start = perf_counter()
        files = raw_paths_by_var[var]
        # raw_ds = xr.open_mfdataset(files, combine="nested", parallel=True)
        # above open_mfdataset approach very slow (time complexity seems at least quadratic in nfiles)
        # instead, just open first and manually concatenate
        raw_ds = xr.open_dataset(files[0])
        for f in files[1:]:
            raw_ds = xr.concat([raw_ds, xr.open_dataset(f)], dim="valid_time")
        print(
            f"Variable {var:<5} ({i+1:02}/{nvars_rel}) took {perf_counter()-start:.2f} s to open {len(files)} files"
        )
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

    # define the independent variable: sample at the chosen point (middle of domain)
    xvar = regdat[idx_var, :, int(latwin / 2) + 1, int(lonwin / 2) + 1]

    # standardize
    xvar = xvar / np.std(xvar)

    print("xvar shape:", xvar.shape)
    print("xvar min,max:", np.min(xvar), np.max(xvar))

    # regress variables
    start = perf_counter()
    regf = np.zeros([nvars_rel, latwin, lonwin])
    for var in range(nvars_rel):
        print(f"regressing variable {var+1:02}/{nvars_rel}: {relevant_vars[var]:<5}") # {var:<5} ({i+1:02}/{nvars_rel})
        for j in range(latwin):
            for i in range(lonwin):
                slope, intercept, r_value, p_value, std_err = linregress(
                    xvar, regdat[var, :, j, i]
                )
                regf[var, j, i] = slope * amp + intercept

        # spatially localize
        regf[var, :] = locfunc[iminlat:imaxlat, iminlon:imaxlon] * regf[var, :]

    stop = perf_counter()
    print(
        f"Regression computation completed in {stop - start:.2f} seconds, averaging {(stop - start) / (n_times):.2f} s/date."
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
            "lat": lat,
            "lon": lon,
        },
    )

    zero_da = xr.DataArray(np.zeros((nlat, nlon)), dims=["lat", "lon"])

    for i, var in enumerate(relevant_vars):
        output_ds[var] = zero_da.copy()
        output_ds[var][
            dict(lat=slice(iminlat, imaxlat), lon=slice(iminlon, imaxlon))
        ] = regf[i]
    for i, var in enumerate(params_invariant):
        output_ds[var] = zero_da.copy()

    if rgfile.exists():
        print(f"Warning: {rgfile} already exists. Overwriting.")
        rgfile.unlink()  # remove the existing file
    output_ds.to_netcdf(rgfile, mode="w", format="NETCDF4")

    print(f"Regression fields saved to {rgfile}")
    
    return output_ds


#
# START: parameters and call to compute_regression
#

# base_dir
base_dir = Path("/N/slate/jmelms/projects/IC")

# set dates
start_end_years_inc = [2015, 2019]
year_range = range(
    start_end_years_inc[0], start_end_years_inc[1] + 1
)  # inclusive range

# collect individual netcdf files here:
rpath = base_dir / "raw"

### DJF ###

ic_months = [12, 1, 2]
ic_str = "DJF"

# write regression results here:
opath = (
    base_dir
    / f"{ic_str}_{start_end_years_inc[0]}-{start_end_years_inc[1]}/reg_pert_files"
)
opath.mkdir(parents=False, exist_ok=True)

# set lat/lon of perturbation in degrees N, E
ylat = 40
xlon = 150
# localization radius in km for the scale of the initial perturbation
locrad = 2000.0
# scaling amplitude for initial condition (1=climo variance at the base point)
amp = -1.0

if (opath/f"{ic_str}_{int(ylat)}N_{int(xlon)}E_z-regression_super.nc").exists():
    print(f"Regression file already exists for {ylat}N, {xlon}E. Skipping computation.")
    super_ds = xr.open_dataset(opath/f"{ic_str}_{int(ylat)}N_{int(xlon)}E_z-regression_super.nc")
else:
    super_ds = compute_regression(
            year_range,
            ic_months,
            ic_str,
            ylat,
            xlon,
            locrad,
            amp,
            rpath,
            opath,
            model="super",
        )
    
models = ["SFNO", "Pangu6", "Pangu6x", "Pangu24", "FuXi", "FuXiShort", "FuXiMedium", "FuXiLong", "FCN3", "GraphCastOperational", "FCN"]
for model in models:
    model_var_names = model_info.MODEL_VARIABLES.get(model)["names"]
    model_ds = super_ds[model_var_names]
    save_file = opath / f"{ic_str}_{int(ylat)}N_{int(xlon)}E_z-regression_{model}.nc"
    if save_file.exists():
        print(f"Warning: {save_file} already exists. Overwriting.")
        save_file.unlink()  # remove the existing file
    model_ds.to_netcdf(save_file, mode="w", format="NETCDF4")
    print(f"Saved {model} regression fields to {save_file}")


### JAS ###

ic_months = [7, 8, 9]
ic_str = "JAS"
# set lat/lon of perturbation in degrees N, E
ylat = 15.0
xlon = 360.0 - 40.0
# localization radius in km for the scale of the initial perturbation
locrad = 1000.0
# scaling amplitude for initial condition (1=climo variance at the base point)
amp = -1.0

# write regression results here:
opath = (
    base_dir
    / f"{ic_str}_{start_end_years_inc[0]}-{start_end_years_inc[1]}/reg_pert_files"
)
opath.mkdir(parents=False, exist_ok=True)

if (opath/f"{ic_str}_{int(ylat)}N_{int(xlon)}E_z-regression_super.nc").exists():
    print(f"Regression file already exists for {ylat}N, {xlon}E. Skipping computation.")
    super_ds = xr.open_dataset(opath/f"{ic_str}_{int(ylat)}N_{int(xlon)}E_z-regression_super.nc")
else:
    super_ds = compute_regression(
            year_range,
            ic_months,
            ic_str,
            ylat,
            xlon,
            locrad,
            amp,
            rpath,
            opath,
            model="super",
        )

models = ["SFNO", "Pangu6", "Pangu6x", "Pangu24", "FuXi", "FuXiShort", "FuXiMedium", "FuXiLong", "FCN3", "GraphCastOperational", "FCN"]
for model in models:
    model_var_names = model_info.MODEL_VARIABLES.get(model)["names"]
    model_ds = super_ds[model_var_names]
    save_file = opath / f"{ic_str}_{int(ylat)}N_{int(xlon)}E_z-regression_{model}.nc"
    if save_file.exists():
        print(f"Warning: {save_file} already exists. Overwriting.")
        save_file.unlink()  # remove the existing file
    model_ds.to_netcdf(save_file, mode="w", format="NETCDF4")
    print(f"Saved {model} regression fields to {save_file}")
#
# END: parameters and call to compute_regression
#
