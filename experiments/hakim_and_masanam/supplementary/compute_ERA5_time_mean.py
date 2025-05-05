"""
This script uses the saved netcdf files from the CDS downloader script to 
compute the time mean for DJF and JJA 0z 1979-2019 at 10 day intervals. 

Will need minor modifications if you used the CDS downloader to download
ERA5 data split up by level. 
"""
import xarray as xr
from pathlib import Path
import os
import numpy as np
from time import perf_counter

def compute_time_mean_from_files(variables: list, months: list, scratch_dir: Path) -> xr.Dataset:
    """
    Computes the time mean for a list of variables and months.
    """
    time_mean_datasets = []
    nvar = len(variables)
    for i, variable in enumerate(variables): 
        print(f"  -> {variable} ({i+1}/{nvar})")
        month_paths = [scratch_dir / f"v={variable}_m={month}.nc" for month in months]
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
        if "pressure_levels" in var_ds.dims:
            chunks["pressure_level"] = -1
        
        var_ds = var_ds.chunk(chunks)
        time_mean_var_ds = var_ds.mean(dim="valid_time").compute()
        time_mean_datasets.append(time_mean_var_ds)

    time_mean_ds = xr.merge(time_mean_datasets).sortby("latitude") # flip latitude index
    
    return time_mean_ds

pl_variables = [
    "geopotential",
    "relative_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind"
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

this_dir = Path(__file__).parent
scratch_dir = Path(os.environ.get("SCRATCH")) / "dcmip" / "era5"
save_dir = this_dir / ".." / "data"
all_vars = pl_variables + sfc_variables
DJF = ["12", "01", "02"]
JJA = ["07", "08", "09"]

seasons = ["JJA"]

if "DJF" in seasons:
    print("Computing time mean for DJF 0z 1979-2019 at 10 day intervals...")
    start = perf_counter()
    DJF_time_mean_ds = compute_time_mean_from_files(all_vars, DJF, scratch_dir)
    print(f"Saving DJF time mean to {save_dir / 'DJF_ERA5_time_mean.nc'}")
    DJF_time_mean_ds.to_netcdf(save_dir / "DJF_ERA5_time_mean.nc")
    end = perf_counter()
    print(f"DJF ran for: {(end - start)/60:.1f} minutes")

if "JJA" in seasons:
    print("Computing time mean for JAS 0z 1979-2019 at 10 day intervals...")
    start = perf_counter()
    JJA_time_mean_ds = compute_time_mean_from_files(all_vars, JJA, scratch_dir)    
    print(f"Saving JJA time mean to {save_dir / 'JJA_ERA5_time_mean.nc'}")
    JJA_time_mean_ds.to_netcdf(save_dir / "JJA_ERA5_time_mean.nc")
    end = perf_counter()
    print(f"JAS ran for: {(end - start)/60:.1f} minutes")