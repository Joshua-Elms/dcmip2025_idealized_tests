import xarray as xr
import datetime as dt
import numpy as np
import calendar
from time import perf_counter
import warnings
import dask
import os
import gc
from dask.distributed import Client, LocalCluster

# Suppress warnings -- dask puts out a lot of them RE: chunking
warnings.filterwarnings("ignore", category=UserWarning, module="dask")

# Setup dask client with memory limit and workers matching CPU cores
n_workers = 4  # Set to the number of available cores
memory_limit = "45GB"  # Per worker memory limit (~180GB total)
scratch_dir = os.environ.get("SCRATCH", "/glade/derecho/scratch/jmelms")  # Default scratch directory

cluster = LocalCluster(
    n_workers=n_workers, 
   threads_per_worker=1, 
   memory_limit=memory_limit,
   local_directory=scratch_dir,
   )
client = Client(cluster)
print(f"Dask dashboard available at: {client.dashboard_link}")

# following Hakim & Masanam (2024) we will use the 0z 1979-2019 DJF time mean
# https://doi.org/10.1175/AIES-D-23-0090.1
start_year = 1979
stop_year = 1979 # 2019
time_utc = 0
months = [12] # DJF
times = []
sfno_levels = [50,100,150,200,250,300,400,500,600,700,850,925,1000]
save_path = "data/DJF_mean_sfno.nc"

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
        
# Define optimal chunk sizes
chunks = {'time': -1, 'level': 13, 'latitude': 720, 'longitude': 1440}

# Process data in batches by variable to avoid memory issues
all_variables = list(pl_variables.keys()) + list(sfc_variables.keys())
time_mean_datasets = []

start_time = perf_counter()
print(f"Starting data processing with {n_workers} workers")

# Process by variable group to conserve memory
for var_group in [pl_variables, sfc_variables]:
    var_names = list(var_group.keys())
    print(f"Processing {var_names}")
    
    # Create separate file lists for each variable
    for var_name, var_code in var_group.items():
        if var_name in pl_variables:
            # Process pressure level variables
            var_files = [f for f in pl_filenames if var_code in f]
            
            # Open only this variable's files with specific chunking
            print(f"Loading {var_name} files...")
            ds_var = xr.open_mfdataset(var_files, parallel=True, combine="by_coords", engine="h5netcdf")
            ds_var = ds_var.chunk(chunks) # reset chunks to preferred for mean computation
            
            # Filter to required levels and compute mean
            if 'level' in ds_var.dims:
                ds_var = ds_var.sel(time=times, level=sfno_levels)
            else:
                ds_var = ds_var.sel(time=times)
                
            # Compute and store this variable's mean
            print(f"Computing mean for {var_name}...")
            mean_var = ds_var.mean(dim="time", keep_attrs=True)
            time_mean_datasets.append(mean_var)
            
            # Explicitly close to free memory
            ds_var.close()
            del ds_var
            
        else:
            # Process surface variables similarly
            var_files = [f for f in sfc_filenames if var_code in f]
            
            print(f"Loading {var_name} files...")
            ds_var = xr.open_mfdataset(var_files, parallel=True, combine="by_coords", 
                                      chunks=chunks, engine="h5netcdf")
            
            ds_var = ds_var.sel(time=times)
            print(f"Computing mean for {var_name}...")
            mean_var = ds_var.mean(dim="time", keep_attrs=True)
            time_mean_datasets.append(mean_var)
            
            ds_var.close()
            del ds_var
            
        # Force garbage collection
        gc.collect()
        
# Combine all variable means into one dataset
print("Combining all variable means...")
time_mean_ds = xr.merge(time_mean_datasets)

# Save the result
print(f"Saving results to {save_path}...")
time_mean_ds.to_netcdf(save_path, mode="w", format="NETCDF4")
stop_time = perf_counter()

print(f"Computed time mean in {stop_time - start_time:.1f} seconds")
print(f"Data at {save_path}")

# Shutdown dask client
client.close()
cluster.close()