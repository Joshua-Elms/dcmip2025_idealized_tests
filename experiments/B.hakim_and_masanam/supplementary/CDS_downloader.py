"""
This script downloads ERA5 at 10-day intervals from 1979-2019 DJF and JAS 0z mean of the necessary
fields to run the SFNO 73 channel model, graphcast_small 83 channel model, and incidentally, the Pangu model.. 

For parallelization, the script uses the multiprocessing library to download
the data in chunks. Each chunk is a time series from 1979-2019 for a single variable
at a single level (sfc or one of the 13 pressure levels). 
"""
import cdsapi
import numpy as np
from pathlib import Path
import os
import multiprocessing as mp
from time import perf_counter

this_dir = Path(__file__).parent
scratch_dir = Path(os.environ.get("SCRATCH")) / "dcmip" / "era5"
ncpus=4 # number of CPUs to use for parallelization, don't exceed ncpus from job request

pl_variables = [ 
    "geopotential",
    "relative_humidity",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]
# # just used for downloading extra pangu/graphcast_small data, temporary
# pl_variables = [
#     "specific_humidity",
#     "vertical_velocity",
# ]
# sfc_variables = []
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

def download_time_chunk(variable, years, month, days, hours_utc, level, dataset, scratch_dir,):
    """
    Downloads a time chunk of data from the CDS API and computes the time mean.
    """
    client = cdsapi.Client()
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
        
    # Download the data
    print(f"DOWNLOADING: {fname} ({years[0]}-{years[-1]})")
    start_time = perf_counter()
    try:
        client.retrieve(dataset, request, scratch_dir / fname)
        
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
        args = (pl_variable, years, month, days, hours_utc, p_levels, pl_dataset, scratch_dir)
        args_list.append(args)

    for variable in sfc_variables:
        args = (variable, years, month, days, hours_utc, "sfc", sfc_dataset, scratch_dir)
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