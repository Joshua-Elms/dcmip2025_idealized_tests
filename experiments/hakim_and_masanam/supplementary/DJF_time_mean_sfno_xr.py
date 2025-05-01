import xarray as xr
import datetime as dt
import numpy as np
import calendar
from time import perf_counter
import warnings

# Suppress warnings -- dask puts out a lot of them RE: chunking
warnings.filterwarnings("ignore", category=UserWarning, module="dask")

# following Hakim & Masanam (2024) we will use the 0z 1979-2019 DJF time mean
# https://doi.org/10.1175/AIES-D-23-0090.1
start_year = 1979
stop_year = 2019 # 2019
time_utc = 0
months = [12, 1, 2] # DJF
times = []
sfno_levels = [50,100,150,200,250,300,400,500,600,700,850,925,1000]
save_path = "data/DJF_mean_sfno_xr.nc"

# get all dates for the DJF months
for year in range(start_year, stop_year + 1):
    for month in months:
        date = dt.datetime(year, month, 1, time_utc)

        while date.month == month: # keep adding days until we reach the next month
            times.append(date)
            date += dt.timedelta(days=1)

times = np.array(times)

# set up variable names and file path templates
e5_base = "/glade/campaign/collections/rda/data/ds633.0"
pl_template = f"{e5_base}/e5.oper.an.pl/{{year:04}}{{month:02}}/e5.oper.an.pl.{{var_code}}.{{year:04}}{{month:02}}{{day:02}}00_{{year:04}}{{month:02}}{{day:02}}23.nc"
pl_variables = {
    "u": "128_131_u.ll025uv",
    "v": "128_132_v.ll025uv",
    "t": "128_130_t.ll025sc",
    "z": "128_129_z.ll025sc",
    "q": "128_133_q.ll025sc",
    "r": "128_157_r.ll025sc",
}
sfc_template = f"{e5_base}/e5.oper.an.sfc/{{year:04}}{{month:02}}/e5.oper.an.sfc.{{var_code}}.{{year:04}}{{month:02}}0100_{{year:04}}{{month:02}}{{dayend:02}}23.nc"
sfc_variables = {
    "u10": "128_165_10u.ll025sc",
    "v10": "128_166_10v.ll025sc",
    "u100": "228_246_100u.ll025sc",
    "v100": "228_247_100v.ll025sc",
    "sp": "128_134_sp.ll025sc",
    "msl": "128_151_msl.ll025sc",
    "tcw": "128_136_tcw.ll025sc",
    "t2m": "128_167_2t.ll025sc",
}

# create list of filenames to download
pl_filenames = []
sfc_filenames = []
for time in times:
    # one file per day, so no worry about duplicates
    for var_name, var_code in pl_variables.items():
        pl_filenames.append(pl_template.format(year=time.year, month=time.month, day=time.day, var_code=var_code))
        
    dayend = calendar.monthrange(time.year, time.month)[1]
    # one file per month, so we will make sure not to repeat
    for var_name, var_code in sfc_variables.items():
        filename = sfc_template.format(year=time.year, month=time.month, dayend=dayend, var_code=var_code)
        if filename not in sfc_filenames:
            sfc_filenames.append(filename)
        

# download each file, computing time mean
start_time = perf_counter()
n_times = len(times)
ds = xr.open_mfdataset([*pl_filenames, *sfc_filenames], parallel=True, combine="by_coords", chunks="auto", engine="h5netcdf")
ds_size_GB = round(ds.nbytes / 1e9)
print(f"Opened {n_times} files ({ds_size_GB} GB) in {start_time - perf_counter():.1f} seconds")

start_time = perf_counter()
time_mean_ds = ds.sel(time=times, level=sfno_levels).squeeze().mean(dim="time")
time_mean_ds.to_netcdf(save_path, mode="w", format="NETCDF4")
stop_time = perf_counter()

print(f"Computed time mean in {stop_time - start_time:.1f} seconds")
print(f"Data at {save_path}")