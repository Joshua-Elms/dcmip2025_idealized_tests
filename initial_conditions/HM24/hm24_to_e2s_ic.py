"""
This file only exists for the diagnostic purpose of ensuring
that the Pangu24 model results from Hakim and Masanam (2024)
are replicable within this Earth2Studio framework. To check,
we convert the original HM24 ICs and perturbations to the E2S
format and then run the experiment, comparing the results
directly against the ones released in the paper. Passing this
test demonstrates that the modeling and visualization framework
is not causing any differences in outcome between E2S and HM24.
"""

from pathlib import Path
import numpy as np
import xarray as xr

exper = "TC"  # ["ETC", "TC"] are options

e2s_IC_path = "/N/slate/jmelms/projects/IC/JAS_2016-2020/IC_files/Pangu24.nc"
hm24_IC_path = "/N/slate/jmelms/projects/IC/JAS_HM24/IC_files/mean_JAS.h5"
output_IC_path = "/N/slate/jmelms/projects/IC/JAS_HM24/IC_files/Pangu24.nc"

e2s_pert_path = "/N/slate/jmelms/projects/IC/JAS_2016-2020/reg_pert_files_ylat=15.0_xlon=320.0/JAS_15N_320E_z-regression_Pangu24.nc"
hm24_pert_path = "/N/slate/jmelms/projects/IC/JAS_HM24/reg_pert_files/hurricane_JAS_15N_40W_regression.h5"
output_pert_path = "/N/slate/jmelms/projects/IC/JAS_HM24/reg_pert_files/JAS_15N_320E_z-regression_Pangu24.nc"

e2s_IC = xr.open_dataset(e2s_IC_path).astype("float64")
hm24_IC = xr.open_dataset(hm24_IC_path).astype("float64")

e2s_pert = xr.open_dataset(e2s_pert_path).astype("float64")
hm24_pert = xr.open_dataset(
    hm24_pert_path, engine="h5netcdf"
)  # error "NetCDF: Start+count exceeds dimension bound" with default engine... this works I guess

# variable ordering in hm24
hm24_pl_vars = ["z", "q", "t", "u", "v"]
hm24_pl_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
hm24_sl_vars = ["msl", "u10m", "v10m", "t2m"]

# output dataset
ic = xr.zeros_like(e2s_IC)
pert = xr.zeros_like(e2s_pert)
output_dat = [ic, pert]
if exper == "ETC":
    hm24_pl_dat = [
        hm24_IC["mean_pl"].to_numpy(),
        np.roll(hm24_pert["regf_pl"].to_numpy(), shift=(129, 505), axis=(-2, -1)),
    ]
    hm24_sl_dat = [
        hm24_IC["mean_sfc"].to_numpy(),
        np.roll(hm24_pert["regf_sfc"].to_numpy(), shift=(129, 505), axis=(-2, -1)),
    ]
elif exper == "TC":
    # current lowest val in reg pert is loc at index (34,37), should be moved to roughly 
    # (300,1280)
    hm24_pl_dat = [
        hm24_IC["mean_pl"].to_numpy(),
        np.roll(hm24_pert["regf_pl"].to_numpy(), shift=(265, 1243), axis=(-2, -1)),
    ]
    hm24_sl_dat = [
        hm24_IC["mean_sfc"].to_numpy(),
        np.roll(hm24_pert["regf_sfc"].to_numpy(), shift=(265, 1243), axis=(-2, -1)),
    ]
writepaths = [output_IC_path, output_pert_path]

for ds, pl_dat, sl_dat, writepath in zip(
    output_dat, hm24_pl_dat, hm24_sl_dat, writepaths
):
    pl_numpy = pl_dat
    sl_numpy = sl_dat

    # insert pl vars from hm24 to e2s
    for v, var in enumerate(hm24_pl_vars):
        for l, lev in enumerate(hm24_pl_levels):
            var_lev = f"{var}{lev}"
            ds[var_lev] = (("time", "lat", "lon"), pl_numpy[np.newaxis, v, l])

    # insert sl vars from hm24 to e2s
    for v, var in enumerate(hm24_sl_vars):
        ds[var] = (("time", "lat", "lon"), sl_numpy[np.newaxis, v])

    # write dataset
    ds.to_netcdf(writepath)
    print(f"Wrote to {writepath}")

means_E2S = {k: v.item() for k, v in dict(e2s_pert.mean()).items()}
means_HM24 = {k: v.item() for k, v in dict(pert.mean()).items()}
print("Mean differences in E2S and HM24:")
for k in means_E2S:
    v1 = means_E2S[k]
    v2 = means_HM24[k]
    print(f"\t{k} = {v1 - v2:0.3f}")
