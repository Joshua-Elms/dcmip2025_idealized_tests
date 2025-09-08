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
import requests


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
        fname = f"v={variable}_l={level}_d={date}.nc"

    elif level == "single":
        fname = f"v={variable}_l=sl_d={date}.nc"

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
        # if we don't do this here, downstream apps
        # get confused by these names; is u10
        # the 10m u wind, or the 10hPa u wind?
        rewrite_vars = { 
            "10m_v_component_of_wind": ("v10", "v10m"),
            "10m_u_component_of_wind": ("u10", "u10m"),
            "100m_v_component_of_wind": ("v100", "v100m"),
            "100m_u_component_of_wind": ("u100", "u100m"),
        }
        if variable in rewrite_vars:
            old_name, new_name = rewrite_vars[variable]
            ds = xr.open_dataset(download_dir / fname)
            ds = ds.rename_vars({old_name: new_name})
            (download_dir / fname).unlink()  # remove original file
            ds.to_netcdf(download_dir / fname)
    except Exception as e:
        print(f"ERROR: request {fname} failed: {e}")
        return fname, "failed"

    return fname, "succeeded"


def run_parallel_download(dates, var_names, var_types, ncpus):
    args_list = []
    for date in dates:
        for var_name, var_type in zip(var_names, var_types):
            if var_type == model_info.PL:
                p_level = int(var_name[1:]) # extract level from var name
                args = (var_name, date, p_level, pl_dataset, raw_data_dir)
            elif var_type == model_info.SL:
                args = (var_name, date, "single", sfc_dataset, raw_data_dir)
            else:
                continue # invariant variable, skip download

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
    var_names: list[str],
    var_types: list[int],
    download_dir: Path,
    save_dir: Path,
) -> xr.Dataset:
    """
    Computes the time mean for a list of variables and months.
    """
    for var_name, var_type in zip(var_names, var_types):
        if var_type == model_info.PL:
            pressure_level = int(var_name[1:])  # extract level from var name
            fpaths = [download_dir / f"v={var_name}_l={pressure_level}_d={date}.nc" for date in dates]
            savepath = save_dir / f"{var_name}_{pressure_level}_tm.nc"
        elif var_type == model_info.SL:
            fpaths = [download_dir / f"v={var_name}_l=sl_d={date}.nc" for date in dates]
            savepath = save_dir / f"{var_name}_sl_tm.nc"
        if savepath.exists():
            print(f"File {savepath} already exists, skipping.")
            continue
        # Open the dataset and compute the time mean
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
        output_path = download_dir / f"v=total_precipitation_06_l=sl_d={chunk[0]}.nc"
        if output_path.exists():
            print(f"File {output_path} already exists, skipping.")
            continue
        fpaths = [download_dir / f"v=total_precipitation_l=sl_d={date}.nc" for date in chunk]
        ds = xr.open_mfdataset(fpaths, combine="nested", parallel=True)
        t = chunk[0]
        # add chunk[0] as valid_time coord
        ds = ds.sum(dim="valid_time")
        ds["tp"] = ds["tp"].expand_dims(
            dict(
                valid_time=[
                    np.datetime64(f"{t[:4]}-{t[4:6]}-{t[6:8]}T{t[8:]}").astype(
                        "datetime64[ns]"
                    )
                ]
            ),
            axis=0,
        )
        ds = ds.rename_vars({"tp": "tp06"})
        # write 6-hourly total precipitation file valid for the end of the interval
        ds.to_netcdf(output_path)


def create_ICs_from_time_means(
    season: str, time_mean_dir: Path, model: str, var_names: list[str], var_types: list[int]
) -> xr.Dataset:
    """Creates initial conditions from the time means for a given season and model."""
    # open all time mean files and concatenate them into a single dataset
    # where each variable is a separate DataArray
    ds = xr.Dataset()
    for variable, var_type in zip(var_names, var_types):
        if var_type == model_info.PL:
            level = int(variable[1:])  # extract level from var name
            new_var_name = f"{lexicon[variable]}{level}"
            file_path = time_mean_dir / f"{variable}_{level}_tm.nc"

        elif var_type == model_info.SL:
            new_var_name = lexicon[variable]
            file_path = time_mean_dir / f"{variable}_sl_tm.nc"
        ds[new_var_name] = xr.open_dataarray(file_path).squeeze(drop=True)
    # make it match the E2S format
    ds = ds.rename(
        {
            "latitude": "lat",
            "longitude": "lon",
        }
    )
    t = [np.datetime64("2000-01-01 00:00")]
    if model in [
        "GraphCastOperational",
        "FuXi",
    ]:  # TODO: parametrize this and automatically adjust
        t = [np.datetime64("1999-12-31 18:00")] + t
    ds = ds.expand_dims({"time": t})
    ds = ds.drop_vars(["number", "expver"], errors="ignore")

    return ds


if __name__ == "__main__":
    ncpus = 4  # number of CPUs to use for parallelization, don't exceed ncpus from job request
    pl_dataset = "reanalysis-era5-pressure-levels"
    sfc_dataset = "reanalysis-era5-single-levels"
    start_end_years_inc = [2018, 2019]
    year_range_name = f"{start_end_years_inc[0]}-{start_end_years_inc[1]}"
    year_range = range(
        start_end_years_inc[0], start_end_years_inc[1] + 1
    )  # inclusive range
    seasons = {"DJF": [12, 1, 2], "JAS": [7, 8, 9]}
    models = ["SFNO", "Pangu6", "GraphCastOperational", "FuXi"]
    static_fields = {
        "geopotential": "https://confluence.ecmwf.int/download/attachments/140385202/geo_1279l4_0.1x0.1.grib2_v4_unpack.nc?version=1&modificationDate=1591983422003&api=v2",
        "land_sea_mask": "https://confluence.ecmwf.int/download/attachments/140385202/lsm_1279l4_0.1x0.1.grb_v4_unpack.nc?version=1&modificationDate=1591983422208&api=v2",
    }
    base_data_dir = Path("/N/slate/jmelms/projects/HM24_ICs")
    raw_data_dir = base_data_dir / "raw"
    for season, months in seasons.items():
        print(f"Processing season {season}")
        dates = generate_dates(year_range, months)
        run_parallel_download(dates, model_info.MASTER_VARIABLE_NAMES, model_info.MASTER_VARIABLES_TYPES, ncpus)
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
                        tp_dates, ["total_precipitation_06"], "single", ncpus
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
            # static fields are weird
            # because they are always single level
            # and only one of them needs to be downloaded
            # so we download them to raw and then stick them
            # in the time_means
            if static_fields:  # check whether any need to be downloaded
                for field, link in static_fields.items():
                    raw_path = raw_data_dir / f"{field}.nc"
                    if not raw_path.exists():
                        print(
                            FileNotFoundError(
                                f"Field {field} not found at {raw_path}, see extract_static_graphcast_fields.py"
                            )
                        )
                    # copy to time_means if needed
                    fpath = time_mean_dir / f"{field}_tm.nc"
                    if fpath.exists():
                        fpath.unlink()
                    fpath.symlink_to(raw_path)
            save_path = IC_output_dir / f"{model}.nc"
            if save_path.exists():
                print(f"File {save_path} already exists, skipping.")
                continue
            IC_ds = create_ICs_from_time_means(
                season, time_mean_dir, model, model_vars, p_levels
            )
            print(f"Saving ICs to {save_path}")
            IC_ds.to_netcdf(save_path)
