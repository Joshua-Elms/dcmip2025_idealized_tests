"""
This script uses the saved netcdf files from the CDS downloader script to 
compute the time mean for DJF and JAS 0z 1979-2019 at 10 day intervals. 

Will need minor modifications if you used the CDS downloader to download
ERA5 data split up by level. 

This would be much more efficient if means for all variables were computed
and merged into one dataset which is subsetted for each model, but this
works with enough job time. 

Note: the first run of this script threw HDF5 errors when reading the 
files for August (08) temperature (t) and 10m u-component of wind (u10).
If you have the same problem, try re-downloading those files from the CDS.
"""
import xarray as xr
from pathlib import Path
import os
from time import perf_counter

name_dict = dict( 
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
        pressure_level="level",
    )

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
        if "pressure_level" in var_ds.dims:
            chunks["pressure_level"] = -1
        
        var_ds = var_ds.chunk(chunks)
        time_mean_var_ds = var_ds.mean(dim="valid_time").compute()
        time_mean_datasets.append(time_mean_var_ds)

    # flip latitude index
    time_mean_ds = xr.merge(time_mean_datasets).sortby("latitude") 
    
    # extra coord from ecmwf, unneeded
    time_mean_ds = time_mean_ds.drop_vars("number") 
    
    # add ensemble dimension and coordinate to data
    time_mean_ds = time_mean_ds.expand_dims({"ensemble": 1}, axis=0)
    
    # add time dimension and coordinate to data
    time_mean_ds = time_mean_ds.expand_dims({"time": 1}, axis=0)
    # time_mean_ds = time_mean_ds.assign_coords({"time": np.array(dt.datetime(1850,1,1))})
    
    # rename to match the variables in the model
    vars_and_coords = list(time_mean_ds.data_vars) + list(time_mean_ds.coords)
    rename_dict = {var: name_dict.get(var) for var in vars_and_coords if var in name_dict}
    print(f"Renaming variables: \n{rename_dict}")
    time_mean_ds = time_mean_ds.rename(rename_dict)
    
    return time_mean_ds

model_variables = dict(
    sfno=[
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
        ],
    pangu=[
        "geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "mean_sea_level_pressure",
        ],
    graphcast_small_mtp06=[
        "geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "mean_sea_level_pressure"
        ],
)

this_dir = Path(__file__).parent
scratch_dir = Path(os.environ.get("SCRATCH")) / "dcmip" / "era5"
save_dir = Path(os.environ.get("SCRATCH")) / "dcmip" / "era5_time_means"
DJF = ["12", "01", "02"]
JAS = ["07", "08", "09"]

seasons = ["DJF", "JAS"]
models = ["sfno", "pangu", "graphcast_small_mtp06"]

for model in models:
    all_vars = model_variables[model]
    print(f"Starting {model} with variables: \n{all_vars}\n\n")
    if "DJF" in seasons:
        print("Computing time mean for DJF 0z 1979-2019 at 10 day intervals...")
        start = perf_counter()
        DJF_time_mean_ds = compute_time_mean_from_files(all_vars, DJF, scratch_dir)
        print(f"Saving DJF time mean to {save_dir / f'DJF_ERA5_time_mean_{model}.nc'}")
        DJF_time_mean_ds.to_netcdf(save_dir / f"DJF_ERA5_time_mean_{model}.nc")
        DJF_time_mean_ds.close()
        end = perf_counter()
        print(f"DJF ran for: {(end - start)/60:.1f} minutes")

    if "JAS" in seasons:
        print("Computing time mean for JAS 0z 1979-2019 at 10 day intervals...")
        start = perf_counter()
        JAS_time_mean_ds = compute_time_mean_from_files(all_vars, JAS, scratch_dir)    
            
        print(f"Saving JAS time mean to {save_dir / f'JAS_ERA5_time_mean_{model}.nc'}")
        JAS_time_mean_ds.to_netcdf(save_dir / f"JAS_ERA5_time_mean_{model}.nc")
        JAS_time_mean_ds.close()
        end = perf_counter()
        print(f"JAS ran for: {(end - start)/60:.1f} minutes")