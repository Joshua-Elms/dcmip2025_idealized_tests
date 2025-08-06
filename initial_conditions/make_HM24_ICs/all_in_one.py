"""
1. Download variables to a directory
2. Compute time mean of chosen seasons (DJF, JAS, etc.) for each variable
3. Select a subset of time mean variables to save into a new dataset for each model IC
"""
import cdsapi
import numpy as np
from pathlib import Path
import multiprocessing as mp
import xarray as xr
from time import perf_counter

raw_data_dir = Path("/N/slate/jmelms/projects/HM24_initial_conditions/raw_data")
raw_data_dir.mkdir(parents=True, exist_ok=True)
time_mean_dir = Path("/N/slate/jmelms/projects/HM24_initial_conditions/time_means")
time_mean_dir.mkdir(parents=True, exist_ok=True)
IC_files_dir = Path("/N/slate/jmelms/projects/HM24_initial_conditions/IC_files")
IC_files_dir.mkdir(parents=True, exist_ok=True)
ncpus=6 # number of CPUs to use for parallelization, don't exceed ncpus from job request

pl_variables = [ 
    "geopotential",
    "relative_humidity",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]
sfc_variables = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "surface_pressure",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "total_column_water_vapour"
    ]
p_levels =  [
    50, 100, 150, 200, 250, 300, 
    400, 500, 600, 700, 850, 925, 
    1000
    ]
years = [str(year) for year in np.arange(1979, 2020)] # str years 1979-2019
months = ["12", "01", "02", "07", "08", "09"] # str months DJF and JAS

# days are at 10-day intervals DJF and JAS
days_by_month = { 
    "12": ["1", "11", "21", "31"],
    "01": ["10", "20", "30"],
    "02": ["9", "19"],
    "07": ["1", "11", "21", "31"],
    "08": ["10", "20", "30"],
    "09": ["9", "19", "29"],
}
hours_utc = ["00:00"] # str hours 0z

pl_dataset = "reanalysis-era5-pressure-levels"
sfc_dataset = "reanalysis-era5-single-levels"

def download_time_chunk(variable, years, month, days, hours_utc, level, dataset, download_dir,):
    """
    Downloads a time chunk of data from the CDS API.
    """
    request = {
        "product_type": ["reanalysis"],
        "variable": [variable],
        "year": years,
        "month": month,
        "day": days,
        "time": hours_utc,
        "data_format": "netcdf",
        "download_format": "unarchived"
    }
    if isinstance(level, int): # if level is int, then it's a single pressure level, e.g. 1000
        request["pressure_level"] = [str(level)] # add to request
        fname = f"v={variable}_p={level}_m={month}.nc"
        
    elif isinstance(level, (list, tuple, np.ndarray)): # if level is list, then it's a list of pressure levels 
        request["pressure_level"] = [str(l) for l in level] # add to request
        fname = f"v={variable}_m={month}.nc"
        
    else: # if level is "sfc", then it's a single level
        fname= f"v={variable}_m={month}.nc"
        
    if (download_dir / fname).exists():
        print(f"File {fname} already exists in {download_dir}, skipping download.")
        return fname, 0
        
    # Download the data
    print(f"DOWNLOADING: {fname} ({years[0]}-{years[-1]})")
    start_time = perf_counter()
    client = cdsapi.Client()
    try:
        client.retrieve(dataset, request, download_dir / fname)
        
    except Exception as e:
        print(f"ERROR: downloading {variable} at {level}: {e}")
        return fname, None
    
    end_time = perf_counter()
    print(f"DOWNLOADED: {fname} ({years[0]}-{years[-1]}) in {end_time - start_time:.2f} seconds")
    
    return fname, end_time - start_time

### create list of all arguments to pass to download_time_chunk (one per variable & month)
args_list = []
for month in months:
    days = days_by_month[month]
    for pl_variable in pl_variables:
        for p_level in p_levels:
            args = (pl_variable, years, month, days, hours_utc, p_level, pl_dataset, raw_data_dir)
            args_list.append(args)

    for variable in sfc_variables:
        args = (variable, years, month, days, hours_utc, "sfc", sfc_dataset, raw_data_dir)
        args_list.append(args)
        
### download the data in parallel
print(f"mp.Pool using ncpus={ncpus}")
print(f"downloading {len(args_list)} files in parallel")

with mp.Pool(processes=ncpus) as pool:
    results = pool.starmap(download_time_chunk, args_list)

### show results
failed_downloads = [fname for fname, time in results if time is None]
failed_downloads_str = '\n\tfname - '.join(failed_downloads) if failed_downloads else 'None'
successful_downloads = [(fname, time) for fname, time in results if time is not None]
avg_times = sum([time for _, time in successful_downloads])/len(successful_downloads)

print(f"Average download time for {len(successful_downloads)} files: {avg_times:.2f} seconds")
print(f"Downloads failed for {len(failed_downloads)} files: \n\t{failed_downloads_str}")





















CDS_to_E2S = dict( 
        u10="u10m",
        v10="v10m",
        u100="u100m",
        v100="v100m",
        t2m="t2m",
        sp="sp",
        msl="msl",
        tcwv="tcwv",
        t="t",
        z="z",
        r="r",
        q="q",
        u="u",
        v="v",
        w="w",
        pressure_level="level",
        latitude="lat",
        longitude="lon",
    )

def compute_time_mean_from_files(sl_variables: list, pl_variables: list, season: str, download_dir: Path, save_dir: Path, pressure_levels: list|None = None) -> xr.Dataset:
    """
    Computes the time mean for a list of variables and months.
    """
    if len(sl_variables) == 0 and len(pl_variables) == 0:
        raise ValueError("At least one of sl_variables or pl_variables must be provided.")
    if len(pl_variables) > 0 and pressure_levels is None:
        raise ValueError("If pl_variables are provided, pressure_levels must also be provided.")
    
    months_to_season = {
        "DJF": ["12", "01", "02"],
        "JAS": ["07", "08", "09"],
    }
    months = months_to_season[season]
    
    nvar = len(sl_variables) + len(pl_variables) * len(pressure_levels)
    # single level variables
    for i, variable in enumerate(sl_variables): 
        savepath = save_dir / f"{variable}_{season}_time_mean.nc"
        if savepath.exists():
            print(f"File {savepath} already exists, skipping.")
            continue
        print(f"  -> {variable} ({i+1}/{nvar})")
        month_paths = [download_dir / f"v={variable}_m={month}.nc" for month in months]
        for month_path in month_paths:
            if not month_path.exists():
                raise FileNotFoundError(f"File {month_path} does not exist.")
            
        var_ds = xr.open_mfdataset(
            month_paths, combine="nested", parallel=True,
            )
        chunks={ # chunking along time to speed up time-mean
                "valid_time": -1, 
                "latitude": 103,
                "longitude": 30,
                }
        
        var_ds = var_ds.chunk(chunks)
        time_mean_var_ds = var_ds.mean(dim="valid_time").compute()
        time_mean_var_ds = time_mean_var_ds.sortby("latitude") # flip latitude index
        time_mean_var_ds = time_mean_var_ds.drop_vars("number") # extra coord from ecmwf, unneeded
        time_mean_var_ds.to_netcdf(savepath)

    # pressure levels variables
    for i, variable in enumerate(pl_variables): 
        for j, pressure_level in enumerate(pressure_levels):
            savepath = save_dir / f"{variable}_{pressure_level}_hPa_{season}_time_mean.nc"
            if savepath.exists():
                print(f"File {savepath} already exists, skipping.")
                continue
            # print expression len(single_levels) + len(pressure_levels) * i + j + 1
            print(f"  -> {variable} at {pressure_level} hPa ({len(sl_variables) + len(pressure_levels) * i + j + 1}/{nvar})")
            month_paths = [download_dir / f"v={variable}_p={pressure_level}_m={month}.nc" for month in months]
            for month_path in month_paths:
                if not month_path.exists():
                    raise FileNotFoundError(f"File {month_path} does not exist.")
                
            var_ds = xr.open_mfdataset(
                month_paths, combine="nested", parallel=True,
                )
            chunks={ # chunking along time to speed up time-mean
                    "valid_time": -1, 
                    "latitude": 103,
                    "longitude": 30,
                    "pressure_level": -1,
                    }
            
            var_ds = var_ds.chunk(chunks)
            time_mean_var_ds = var_ds.mean(dim="valid_time").compute()
            time_mean_var_ds = time_mean_var_ds.sortby("latitude") # flip latitude index
            time_mean_var_ds = time_mean_var_ds.drop_vars("number") # extra coord from ecmwf, unneeded
            time_mean_var_ds.to_netcdf(savepath)



DJF = ["12", "01", "02"]
JAS = ["07", "08", "09"]

seasons = ["DJF", "JAS"]

all_variables = [
    "geopotential",
    "relative_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "surface_pressure",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "total_column_water_vapour",
    "specific_humidity",
    "vertical_velocity",
    ]

DJF_time_mean_ds = compute_time_mean_from_files(sfc_variables, pl_variables, "DJF", raw_data_dir, time_mean_dir, pressure_levels=p_levels)
JAS_time_mean_ds = compute_time_mean_from_files(sfc_variables, pl_variables, "JAS", raw_data_dir, time_mean_dir, pressure_levels=p_levels)


def create_ICs_from_time_means(season: str, time_mean_dir: Path, model: str, p_levels: list) -> xr.Dataset:
    """Creates initial conditions from the time means for a given season and model.
    """
    models = ["SFNO", "Pangu6", "GraphCastOperational"]
    if model not in models:
        raise ValueError(f"Model {model} not recognized. Must be one of {models}.")
    
    pl_variables, sl_variables = model_variables[model]
    print(f"Creating ICs for {model} for season {season} with {len(pl_variables)} pressure level variables and {len(sl_variables)} single level variables.")
    # open all time mean files and concatenate them into a single dataset
    # where each variable is a separate DataArray
    ds = xr.Dataset()
    for variable in pl_variables:
        for pressure_level in p_levels:
            file_path = time_mean_dir / f"{variable}_{pressure_level}_hPa_{season}_time_mean.nc"
            new_var_name = f"{lexicon[variable]}{pressure_level}"
            ds[new_var_name] = xr.open_dataarray(file_path).squeeze(drop=True) # drop pressure_level dim since it's a single level
            
    for variable in sl_variables:
        file_path = time_mean_dir / f"{variable}_{season}_time_mean.nc"
        ds[lexicon[variable]] = xr.open_dataarray(file_path).squeeze(drop=True)
        
    # make it match the E2S format
    ds = ds.rename({
        "latitude": "lat",
        "longitude": "lon",
    })
    if ds.lat[0] < ds.lat[-1]: # should be in descending order
        ds = ds.sortby("lat", ascending=False)  # flip latitude index if needed
        
    if "time" not in ds.dims:  # if time is not a dimension, add it
        ds = ds.expand_dims({"time": [np.datetime64("2000-01-01")]})  # add a dummy time dimension
        
    return ds

model_variables = dict(
    SFNO=[
        ["geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",],
        [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "mean_sea_level_pressure",
        "surface_pressure",
        "100m_u_component_of_wind",
        "100m_v_component_of_wind",
        "total_column_water_vapour",
        ]],
    Pangu6=[
        ["geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind"],
        ["10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "mean_sea_level_pressure",
        ]],
    GraphCastOperational=[
        ["geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity"],
        ["10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "mean_sea_level_pressure"
        ]],
)

lexicon = {
    "geopotential": "z",
    "relative_humidity": "r",
    "specific_humidity": "q",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "vertical_velocity": "w",
    "10m_u_component_of_wind": "u10m",
    "10m_v_component_of_wind": "v10m",
    "2m_temperature": "t2m",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "100m_u_component_of_wind": "u100m",
    "100m_v_component_of_wind": "v100m",
    "total_column_water_vapour": "tcwv",
}

models = ["SFNO", "Pangu6", "GraphCastOperational"]
for model in models:
    for season in seasons:
        IC_ds = create_ICs_from_time_means(season, time_mean_dir, model, p_levels)
        save_path = IC_files_dir / f"{model}_{season}_IC.nc"
        if save_path.exists():
            print(f"File {save_path} already exists, skipping.")
            continue
        print(f"Saving ICs to {save_path}")
        IC_ds.to_netcdf(save_path)