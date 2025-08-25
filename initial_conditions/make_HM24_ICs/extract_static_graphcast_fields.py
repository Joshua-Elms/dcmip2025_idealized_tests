import requests
import xarray as xr
from pathlib import Path

url = "https://storage.googleapis.com/dm_graphcast/graphcast/dataset/source-era5_date-2022-01-01_res-0.25_levels-13_steps-01.nc"
raw_data_dir = Path("/N/slate/jmelms/projects/HM24_ICs/raw")
lsm_fname = "land_sea_mask.nc"
geopotential_fname = "geopotential.nc"

content = requests.get(url).content
ds = xr.open_dataset(content)[["land_sea_mask", "geopotential_at_surface"]]
ds = ds.rename({"lat": "latitude", "lon": "longitude", "geopotential_at_surface": "geopotential"})
ds["land_sea_mask"].to_netcdf(raw_data_dir / lsm_fname)
print(f"Wrote {raw_data_dir / lsm_fname}")
ds["geopotential"].to_netcdf(raw_data_dir / geopotential_fname)
print(f"Wrote {raw_data_dir / geopotential_fname}")