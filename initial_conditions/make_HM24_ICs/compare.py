import xarray as xr

ds1_path = "/N/slate/jmelms/projects/dcmip2025_idealized_tests/initial_conditions/ic_unperturbed.nc"
ds2_path = "/N/slate/jmelms/projects/HM24_initial_conditions/IC_files/sfno_DJF_IC.nc"

ds1 = xr.open_dataset(ds1_path)
ds2 = xr.open_dataset(ds2_path)

print(ds1)


print(ds2)

breakpoint()