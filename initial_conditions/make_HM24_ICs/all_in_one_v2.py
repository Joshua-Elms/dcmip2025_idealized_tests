"""
1. Download variables to a directory
2. Compute time mean of chosen seasons (DJF, JAS, etc.) for each variable
3. Select a subset of time mean variables to save into a new dataset for each model IC
"""
import cdsapi
from calendar import month, monthrange
import datetime
import numpy as np
from pathlib import Path
import multiprocessing as mp
import xarray as xr

def generate_dates(years: range, months: list[int]):
    """
    Generate dates for the given years and months, incrementing by 10 days.
    Handles leap years for February.
    """
    # input validation
    if not years or not months:
        raise ValueError("Years and months lists must not be empty.")
    if not all(isinstance(year, int) for year in years):
        raise TypeError("All years must be integers.")
    if not all(isinstance(month, int) for month in months):
        raise TypeError("All months must be integers.")
    if not all(1 <= month <= 12 for month in months):
        raise ValueError("All months must be between 1 and 12.")
    ordered = True
    if sorted(months) != months:
        if 12 in months and 1 in months:
            if months.index(12) != (months.index(1) - 1):
                ordered = False
        else:
            ordered = False
    if not ordered:
        raise ValueError("Months must be in ascending order except for December->January.")
    dates = []
    for year in years:
        start_date = datetime.date(year, months[0], 1)
        # figure out end_date
        if months[-1] < months[0]:
            end_year = year + 1
        else:
            end_year = year
        end_month = months[-1]
        end_day = monthrange(end_year, end_month)[1]
        end_date = datetime.date(end_year, end_month, end_day)
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date.strftime("%Y%m%d%H"))
            current_date += datetime.timedelta(days=10)
    return dates

def download_chunk(variable: str, date: str, level: int|str, dataset: str, download_dir: Path):
    """
    Downloads a single timestep of data for one variable from the CDS API.
    """
    request = {
        "product_type": ["reanalysis"],
        "variable": [variable],
        "year": date[0:4],
        "month": date[4:6],
        "day": date[6:8],
        "time": date[8:10] + ":00",
        "data_format": "netcdf",
        "download_format": "unarchived"
    }
    if isinstance(level, int):
        request["pressure_level"] = [str(level)] # add to request
        fname = f"v={variable}_p={level}_d={date}.nc"
           
    elif level == "single":
        fname= f"v={variable}_m={month}.nc"
        
    if (download_dir / fname).exists():
        print(f"File {fname} already exists, skipping download.")
        return fname, "skipped"
        
    # Download the data
    client = cdsapi.Client()
    try:
        client.retrieve(dataset, request, download_dir / fname)
        
    except Exception as e:
        print(f"ERROR: request {fname} failed: {e}")
        return fname, "failed"
    
    return fname, "succeeded"

def compute_time_mean_from_files(dates: list[str], pl_variables: list[str], sl_variables: list[str], pressure_levels: list[int],download_dir: Path, save_dir: Path) -> xr.Dataset:
    """
    Computes the time mean for a list of variables and months.
    """
    if pl_variables is None and sl_variables is None:
        raise ValueError("At least one of pl_variables or sl_variables must be provided.")

    for pl_var in pl_variables:
        for pressure_level in pressure_levels:
            fpaths = [download_dir / f"v={pl_var}_p={pressure_level}_d={date}.nc" for date in dates]
            savepath = save_dir / f"{pl_var}_{pressure_level}_hPa_tm.nc"
            # Open the dataset and compute the time mean
            var_ds = xr.open_mfdataset(fpaths, combine="nested", parallel=True)
            time_mean_var_ds = var_ds.mean(dim="valid_time").compute()
            time_mean_var_ds = time_mean_var_ds.sortby("latitude")  # flip latitude index
            time_mean_var_ds = time_mean_var_ds.drop_vars("number")  # extra coord from ecmwf, unneeded
            time_mean_var_ds.to_netcdf(savepath)

    for sl_var in sl_variables:
        fpaths = [download_dir / f"v={sl_var}_d={date}.nc" for date in dates]
        savepath = save_dir / f"{sl_var}_tm.nc"
        # Open the dataset and compute the time mean
        var_ds = xr.open_mfdataset(fpaths, combine="nested", parallel=True)
        time_mean_var_ds = var_ds.mean(dim="valid_time").compute()
        time_mean_var_ds = time_mean_var_ds.sortby("latitude")  # flip latitude index
        time_mean_var_ds = time_mean_var_ds.drop_vars("number")  # extra coord from ecmwf, unneeded
        time_mean_var_ds.to_netcdf(savepath)

    # # single level variables
    # for i, variable in enumerate(sl_vars): 
    #     savepath = save_dir / f"{variable}_{season}_time_mean.nc"
    #     if savepath.exists():
    #         print(f"File {savepath} already exists, skipping.")
    #         continue
    #     print(f"  -> {variable} ({i+1}/{nvar})")
    #     month_paths = [download_dir / f"v={variable}_m={month}.nc" for month in months]
    #     for month_path in month_paths:
    #         if not month_path.exists():
    #             raise FileNotFoundError(f"File {month_path} does not exist.")
            
    #     var_ds = xr.open_mfdataset(
    #         month_paths, combine="nested", parallel=True,
    #         )
    #     chunks={ # chunking along time to speed up time-mean
    #             "valid_time": -1, 
    #             "latitude": 103,
    #             "longitude": 30,
    #             }
        
    #     var_ds = var_ds.chunk(chunks)
    #     time_mean_var_ds = var_ds.mean(dim="valid_time").compute()
    #     time_mean_var_ds = time_mean_var_ds.sortby("latitude") # flip latitude index
    #     time_mean_var_ds = time_mean_var_ds.drop_vars("number") # extra coord from ecmwf, unneeded
    #     time_mean_var_ds.to_netcdf(savepath)

    # # pressure levels variables
    # for i, variable in enumerate(pl_variables): 
    #     for j, pressure_level in enumerate(pressure_levels):
    #         savepath = save_dir / f"{variable}_{pressure_level}_hPa_{season}_time_mean.nc"
    #         if savepath.exists():
    #             print(f"File {savepath} already exists, skipping.")
    #             continue
    #         # print expression len(single_levels) + len(pressure_levels) * i + j + 1
    #         print(f"  -> {variable} at {pressure_level} hPa ({len(sl_variables) + len(pressure_levels) * i + j + 1}/{nvar})")
    #         month_paths = [download_dir / f"v={variable}_p={pressure_level}_m={month}.nc" for month in months]
    #         for month_path in month_paths:
    #             if not month_path.exists():
    #                 raise FileNotFoundError(f"File {month_path} does not exist.")
                
    #         var_ds = xr.open_mfdataset(
    #             month_paths, combine="nested", parallel=True,
    #             )
    #         chunks={ # chunking along time to speed up time-mean
    #                 "valid_time": -1, 
    #                 "latitude": 103,
    #                 "longitude": 30,
    #                 "pressure_level": -1,
    #                 }
            
    #         var_ds = var_ds.chunk(chunks)
    #         time_mean_var_ds = var_ds.mean(dim="valid_time").compute()
    #         time_mean_var_ds = time_mean_var_ds.sortby("latitude") # flip latitude index
    #         time_mean_var_ds = time_mean_var_ds.drop_vars("number") # extra coord from ecmwf, unneeded
    #         time_mean_var_ds.to_netcdf(savepath)

if __name__ == "__main__":
    
    ncpus=1 # number of CPUs to use for parallelization, don't exceed ncpus from job request
    
    download_tp06 = True

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
        "total_column_water_vapour",
        ]
    p_levels =  [
        50, 100, 150, 200, 250, 300, 
        400, 500, 600, 700, 850, 925, 
        1000
        ]
    pl_dataset = "reanalysis-era5-pressure-levels"
    sfc_dataset = "reanalysis-era5-single-levels"
    start_end_years_inc = [1979, 2019]
    year_range_name = f"{start_end_years_inc[0]}-{start_end_years_inc[1]}"
    year_range = range(start_end_years_inc[0], start_end_years_inc[1] + 1)  # inclusive range
    seasons = {
        "DJF": [12, 1, 2],
        "JAS": [7, 8, 9]
    }
    base_data_dir = Path("/N/slate/jmelms/projects/HM24_ICs")
    raw_data_dir = base_data_dir / "raw"
    for season, months in seasons.items():
        print(f"Processing season {season}")
        dates = generate_dates(year_range, months)
        args_list = []
        for date in dates:
            for pl_variable in pl_variables:
                for p_level in p_levels:
                    args = (pl_variable, date, p_level, pl_dataset, raw_data_dir)
                    args_list.append(args)

            for variable in sfc_variables:
                args = (variable, date, "sfc", sfc_dataset, raw_data_dir)
                args_list.append(args)

        ### download the data in parallel
        print(f"mp.Pool using ncpus={ncpus}")
        print(f"downloading {len(args_list)} files in parallel")

        with mp.Pool(processes=ncpus) as pool:
            results = pool.starmap(download_chunk, args_list)
            
        failed_downloads = [fname for fname, status in results if status is "failed"]
        skipped_downloads = [fname for fname, status in results if status is "skipped"]
        succeeded_downloads = [fname for fname, status in results if status is "succeeded"]

        failed_downloads_str = '\n\tfname - '.join(failed_downloads) if failed_downloads else 'None'
        print(f"Download stats:")
        print(f"\tFailed: {failed_downloads_str}")
        print(f"\tSkipped: {skipped_downloads}")
        print(f"\tSucceeded: {succeeded_downloads}")
        if failed_downloads:
            print(f"\tFailed downloads were: {failed_downloads}")

        save_dir = base_data_dir / "time_means" / f"{season}_{year_range_name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        compute_time_mean_from_files(dates, pl_variables, sfc_variables, p_levels, raw_data_dir, save_dir)
