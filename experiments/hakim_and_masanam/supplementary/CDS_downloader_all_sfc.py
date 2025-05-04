"""
This script downloads ERA5 fields and computes the 1979-2019 DJF 0z mean of the necessary
fields to run the SFNO 73 channel model. 

For parallelization, the script uses the multiprocessing library to download
the data in chunks. Each chunk is a time series from 1979-2019 for a single variable
at a single level (sfc or one of the 13 pressure levels). 

The time-mean of the chunk is computed and saved to a netcdf file, then all chunks are 
concatenated into a single netcdf file at the end. 
"""

import xarray as xr
import cdsapi
import datetime as dt
import numpy as np
from pathlib import Path
import os
import multiprocessing as mp
from utils import inference
from time import perf_counter

this_dir = Path(__file__).parent
scratch_dir = Path(os.environ.get("SCRATCH")) / "dcmip" / "tmp"
ncpus=4 # or mp.cpu_count()

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
months = ["12", "01", "02"] # str months DJF
days = [str(day) for day in np.arange(1, 32)] # str days 1-31 (<31-day months will use fewer)
hours_utc = ["00:00"] # str hours 0z

sfc_dataset = "reanalysis-era5-single-levels"

def download_time_chunk(variable, years, months, days, hours_utc, level, dataset, scratch_dir,):
    """
    Downloads a time chunk of data from the CDS API and computes the time mean.
    """
    client = cdsapi.Client()
    request = {
        "product_type": ["reanalysis"],
        "variable": [variable],
        "year": years,
        "month": months,
        "day": days,
        "time": hours_utc,
        "data_format": "netcdf",
        "download_format": "unarchived"
    }
    if isinstance(level, int): # if level is int, then it's a pressure level, e.g. 1000
        request["pressure_level"] = [str(level)] # add to request
    
    # Download the data
    start_time = perf_counter()
    try:
        client.retrieve(dataset, request, f"{scratch_dir}/{variable}_{level}.nc")
        
    except Exception as e:
        print(f"Error downloading {variable} at {level}: {e}")
    
    # Load the data into an xarray dataset
    ds = xr.open_dataarray(f"{scratch_dir}/{variable}_{level}.nc").squeeze()
    ds = ds.sortby(ds.latitude) # latitude is reversed in the CDS data
    
    # Compute the time mean
    ds_mean = ds.mean(dim="valid_time")
    # add singleton variable dimension
    var_name = f"{variable}_{level}" if isinstance(level, int) else variable
    ds_mean["variable"] = var_name

    # Save the time mean to a netcdf file
    fname = f"{variable}_{level}_mean.nc"
    fpath = scratch_dir / fname
    ds_mean.to_netcdf(fpath)
    
    ds.close()
    ds_mean.close()
    end_time = perf_counter()
    print(f"Downloaded {fname} ({years[0]}-{years[-1]}) in {end_time - start_time:.2f} seconds")
    
    return fname, end_time - start_time

### create list of all arguments to pass to download_time_chunk (one per variable & level)

args_list = []
for variable in sfc_variables:
    args = (variable, years, months, days, hours_utc, "sfc", sfc_dataset, scratch_dir)
    args_list.append(args)
    
### download the data in parallel

with mp.Pool(processes=ncpus) as pool:
    results = pool.starmap(download_time_chunk, args_list)
    

# ### concatenate the results into a single netcdf file

# time_mean_ds = inference.create_empty_sfno_ds()
# # pl variables
# for variable in pl_variables:
#     var_da_list = []
#     for p_level in p_levels:
#         fpath = scratch_dir / f"{variable}_{p_level}_mean.nc"
#         assert fpath.exists(), f"File {fpath} does not exist"
#         da = xr.open_dataarray(fpath)
#         var_da_list.append(da)
#     var_da = xr.concat(var_da_list, dim="pressure_level")
#     time_mean_ds[variable] = var_da

# for variable in sfc_variables:
#     fpath = scratch_dir / f"{variable}_mean.nc"
#     assert fpath.exists(), f"File {fpath} does not exist"
#     da = xr.open_dataarray(fpath)
#     time_mean_ds[variable] = da
    
# # save the time mean dataset to a netcdf file
# time_mean_ds.to_netcdf(save_path)

# single file download example
# fname, duration = download_time_chunk(**kwargs_list[0])
