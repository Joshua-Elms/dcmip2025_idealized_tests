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
        raise ValueError(
            "Months must be in ascending order except for December->January."
        )
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


def generate_tp06_dates(dates: list[str]) -> list[str]:
    """Return a list of the 6 hours preceding each date in the input list."""
    tp06_dates = []
    for date in dates:
        dt = datetime.datetime.strptime(date, "%Y%m%d%H")
        # don't change the reverse ordering below
        # func "aggregate_tp_files" depends on it
        for i in range(6):
            tp06_dates.append((dt - datetime.timedelta(hours=i)).strftime("%Y%m%d%H"))
    return tp06_dates


def download_chunk(
    variable: str, date: str, level: int | str, dataset: str, download_dir: Path
):
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
        "download_format": "unarchived",
    }
    if isinstance(level, int):
        request["pressure_level"] = [str(level)]  # add to request
        fname = f"v={variable}_p={level}_d={date}.nc"

    elif level == "single":
        fname = f"v={variable}_d={date}.nc"

    else:
        print(
            f"ERROR: level must be int or 'single', got level '{level}' of type {type(level)} instead."
        )
        return f"invalid_level_{level}", "failed"

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


def run_parallel_download(dates, pl_variables, sl_variables, p_levels, ncpus):
    if pl_variables == [] and sl_variables == []:
        raise ValueError(
            "At least one of pl_variables or sl_variables must be provided."
        )
    args_list = []
    for date in dates:
        for pl_variable in pl_variables:
            for p_level in p_levels:
                args = (pl_variable, date, p_level, pl_dataset, raw_data_dir)
                args_list.append(args)

        for variable in sl_variables:
            args = (variable, date, "single", sfc_dataset, raw_data_dir)
            args_list.append(args)

    # download the data in parallel
    print(f"mp.Pool using ncpus={ncpus}")
    print(f"downloading {len(args_list)} files in parallel")

    with mp.Pool(processes=ncpus) as pool:
        results = pool.starmap(download_chunk, args_list)

    failed_downloads = [fname for fname, status in results if status == "failed"]
    skipped_downloads = [fname for fname, status in results if status == "skipped"]
    succeeded_downloads = [fname for fname, status in results if status == "succeeded"]

    failed_downloads_str = (
        "\n\tfname - ".join(failed_downloads) if failed_downloads else "None"
    )
    print(f"Download stats:")
    print(f"\tFailed: {failed_downloads_str}")
    print(f"\tSkipped: {skipped_downloads}")
    print(f"\tSucceeded: {succeeded_downloads}")
    if failed_downloads:
        print(f"\tFailed downloads were: {failed_downloads}")

    return results


def compute_time_mean_from_files(
    dates: list[str],
    pl_variables: list[str],
    sl_variables: list[str],
    pressure_levels: list[int],
    download_dir: Path,
    save_dir: Path,
) -> xr.Dataset:
    """
    Computes the time mean for a list of variables and months.
    """
    if pl_variables == [] and sl_variables == []:
        raise ValueError(
            "At least one of pl_variables or sl_variables must be provided."
        )
    for pl_var in pl_variables:
        for pressure_level in pressure_levels:
            fpaths = [
                download_dir / f"v={pl_var}_p={pressure_level}_d={date}.nc"
                for date in dates
            ]
            savepath = save_dir / f"{pl_var}_{pressure_level}_hPa_tm.nc"
            if savepath.exists():
                print(f"File {savepath} already exists, skipping.")
                continue
            # Open the dataset and compute the time mean
            var_ds = xr.open_mfdataset(fpaths, combine="nested", parallel=True)
            time_mean_var_ds = var_ds.mean(dim="valid_time").compute()
            time_mean_var_ds = time_mean_var_ds.sortby(
                "latitude"
            )  # flip latitude index
            time_mean_var_ds = time_mean_var_ds.drop_vars(
                "number"
            )  # extra coord from ecmwf, unneeded
            time_mean_var_ds.to_netcdf(savepath)

    for sl_var in sl_variables:
        fpaths = [download_dir / f"v={sl_var}_d={date}.nc" for date in dates]
        savepath = save_dir / f"{sl_var}_tm.nc"
        if savepath.exists():
            print(f"File {savepath} already exists, skipping.")
            continue
        # Open the dataset and compute the time mean
        if sl_var == "total_precipitation_06":
            breakpoint()
        var_ds = xr.open_mfdataset(fpaths, combine="by_coords", parallel=True)
        time_mean_var_ds = var_ds.mean(dim="valid_time").compute()
        time_mean_var_ds = time_mean_var_ds.drop_vars(
            ["number", "expver"]
        )  # extra coord from ecmwf, unneeded
        time_mean_var_ds.to_netcdf(savepath)


def aggregate_tp_files(tp_dates: list[str], download_dir: Path):
    """
    Aggregates the hourly total precipitation files for the given dates into 6-hourly total precipitation files
    and writes those completed fields to the same location.
    """
    # divide tp_dates into 6-hourly intervals
    tp_dates_6chunked = [tp_dates[i : i + 6] for i in range(0, len(tp_dates), 6)]
    for chunk in tp_dates_6chunked:
        output_path = download_dir / f"v=total_precipitation_06_d={chunk[0]}.nc"
        if output_path.exists():
            print(f"File {output_path} already exists, skipping.")
            continue
        fpaths = [download_dir / f"v=total_precipitation_d={date}.nc" for date in chunk]
        ds = xr.open_mfdataset(fpaths, combine="nested", parallel=True)
        ds = ds.sum(dim="valid_time").compute()
        # add chunk[0] as valid_time coord
        ds = ds.assign_coords(valid_time=[chunk[0]])
        # write 6-hourly total precipitation file valid for the end of the interval
        ds.to_netcdf(output_path)


def create_ICs_from_time_means(
    season: str, time_mean_dir: Path, model: str, model_variables: dict, p_levels: list
) -> xr.Dataset:
    """Creates initial conditions from the time means for a given season and model."""
    pl_variables, sl_variables = model_variables
    print(
        f"Creating ICs for {model} for season {season} with {len(pl_variables)} pressure level variables and {len(sl_variables)} single level variables."
    )
    # open all time mean files and concatenate them into a single dataset
    # where each variable is a separate DataArray
    ds = xr.Dataset()
    for variable in pl_variables:
        for pressure_level in p_levels:
            file_path = time_mean_dir / f"{variable}_{pressure_level}_hPa_tm.nc"
            new_var_name = f"{lexicon[variable]}{pressure_level}"
            ds[new_var_name] = xr.open_dataarray(file_path).squeeze(
                drop=True
            )  # drop pressure_level dim since it's a single level
    for variable in sl_variables:
        file_path = time_mean_dir / f"{variable}_tm.nc"
        ds[lexicon[variable]] = xr.open_dataarray(file_path).squeeze(drop=True)
    # make it match the E2S format
    ds = ds.rename(
        {
            "latitude": "lat",
            "longitude": "lon",
        }
    )
    # if time is not a dimension, add it
    if "time" not in ds.dims:  
        ds = ds.expand_dims(
            {"time": [np.datetime64("2000-01-01")]}
        )  # add a dummy time dimension

    return ds


if __name__ == "__main__":
    ncpus = 4  # number of CPUs to use for parallelization, don't exceed ncpus from job request
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
    model_variables = dict(
        SFNO=[
            [
                "geopotential",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ],
            [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
                "mean_sea_level_pressure",
                "surface_pressure",
                "100m_u_component_of_wind",
                "100m_v_component_of_wind",
                "total_column_water_vapour",
            ],
        ],
        Pangu6=[
            [
                "geopotential",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ],
            [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
                "mean_sea_level_pressure",
            ],
        ],
        GraphCastOperational=[
            [
                "geopotential",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
                "vertical_velocity",
            ],
            [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
                "mean_sea_level_pressure",
                "total_precipitation_06",
            ],
        ],
        FuXi=[
            [
                "geopotential",
                "relative_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ],
            [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
                "mean_sea_level_pressure",
                "total_precipitation_06",
            ],
        ],
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
        "total_precipitation_06": "tp06",
    }
    p_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    # p_levels = [50, 1000]  # debug only
    pl_dataset = "reanalysis-era5-pressure-levels"
    sfc_dataset = "reanalysis-era5-single-levels"
    start_end_years_inc = [2018, 2019]
    year_range_name = f"{start_end_years_inc[0]}-{start_end_years_inc[1]}"
    year_range = range(
        start_end_years_inc[0], start_end_years_inc[1] + 1
    )  # inclusive range
    seasons = {"DJF": [12, 1, 2], "JAS": [7, 8, 9]}
    models = ["SFNO", "Pangu6", "GraphCastOperational", "FuXi"]
    base_data_dir = Path("/N/slate/jmelms/projects/HM24_ICs")
    raw_data_dir = base_data_dir / "raw"
    IC_files_dir = base_data_dir / "IC_files"
    for season, months in seasons.items():
        print(f"Processing season {season}")
        dates = generate_dates(year_range, months)
        run_parallel_download(dates, pl_variables, sfc_variables, p_levels, ncpus)
        save_dir = base_data_dir / f"{season}_{year_range_name}"
        time_mean_dir = save_dir / "time_means"
        IC_output_dir = save_dir / "IC_files"
        for dir in [time_mean_dir, IC_output_dir]:
            dir.mkdir(parents=True, exist_ok=True)
        compute_time_mean_from_files(
            dates, pl_variables, sfc_variables, p_levels, raw_data_dir, time_mean_dir
        )
        for model in models:
            model_vars = model_variables[model]
            # if no model uses tp06, then we won't download it
            # if at least one does, the raw tp fields are downloaded
            # then merged into raw tp06 fields, and handled normally
            # in the "compute_time_mean_from_files" func downstream
            downloaded_tp_raw = False
            if "total_precipitation_06" in model_vars[1]:
                if not downloaded_tp_raw:
                    tp_dates = generate_tp06_dates(dates)
                    run_parallel_download(
                        tp_dates, [], ["total_precipitation"], "single", ncpus
                    )
                    downloaded_tp_raw = True
                    aggregate_tp_files(tp_dates, raw_data_dir)
                    compute_time_mean_from_files(
                        dates,
                        [],
                        ["total_precipitation_06"],
                        "single",
                        raw_data_dir,
                        time_mean_dir,
                    )
            save_path = IC_output_dir / f"{model}.nc"
            if save_path.exists():
                print(f"File {save_path} already exists, skipping.")
                continue
            IC_ds = create_ICs_from_time_means(
                season, time_mean_dir, model, model_vars, p_levels
            )
            print(f"Saving ICs to {save_path}")
            IC_ds.to_netcdf(save_path)
