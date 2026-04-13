import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from utils import model_info

ETC_pert = xr.open_dataset(
    "/N/slate/jmelms/projects/IC/DJF_1979-2019/reg_pert_files_ylat=40_xlon=150/DJF_40N_150E_z-regression_Pangu24.nc"
)
e2s_pert = xr.open_dataset(
    "/N/slate/jmelms/projects/IC/JAS_1979-2019/reg_pert_files_ylat=15.0_xlon=320.0/JAS_15N_320E_z-regression_Pangu24.nc"
)
hm24_pert = xr.open_dataset(
    "/N/slate/jmelms/projects/IC/JAS_HM24/reg_pert_files/JAS_15N_320E_z-regression_Pangu24.nc",
    engine="h5netcdf",
)

diff_pert_max = {
    k: v.values.item() for k, v in dict(np.abs(e2s_pert - hm24_pert).max()).items()
}
diff_pert = (e2s_pert - hm24_pert).isel(lat=slice(270, 325), lon=slice(1255, 1310))


print("Plotting msl for both E2S and HM24 Perturbations")
fig, ax = plt.subplot_mosaic("A\nB\nC", height_ratios=[0.45, 0.1, 0.45], figsize=(6, 8))
ax["A"].imshow(e2s_pert["msl"].values.squeeze())
ax["A"].set_title("E2S")
im = ax["C"].imshow(hm24_pert["msl"].values.squeeze())
ax["C"].set_title("HM24")
fig.colorbar(im, cax=ax["B"], label="Pa", orientation="horizontal")
fig.suptitle("MSL Perturbations")
fig.tight_layout()
fig.savefig("msl_fields_both_pert.png", dpi=300)
fig.clf()
plt.close()

# vars = e2s_pert.data_vars
# for var in vars:
#     fig, ax = plt.subplots()
#     im = ax.imshow(diff_pert[var].values.squeeze())
#     ax.set_title(f"{var}: E2S ({e2s_pert[var].values.min():0.2f}) - HM24 ({hm24_pert[var].values.min():0.2f})")
#     fig.colorbar(im, ax=ax)
#     fig.tight_layout()
#     fig.savefig(f"{var}_diff.png", dpi=300)
#     fig.clf()
#     plt.close()
    
### plot stuve diagram for both perturbations

levels = model_info.STANDARD_13_LEVELS
plot_name = "Stuve Comparison"
fig, ax = plt.subplots()
ds_list = [ETC_pert, e2s_pert, hm24_pert]
data_sources = ["ETC", "E2S", "HM24"]
for i, tds in enumerate(ds_list):
    if data_sources[i] == "ETC":
        tds_subset = tds.isel(lat=slice(160, 240), lon=slice(560, 640))
    else:
        tds_subset = tds.isel(lat=slice(270, 325), lon=slice(1255, 1310))
    vert_t_profile = []
    for lev in levels:
        vert_t_profile.append(tds_subset[f"t{lev}"].values.mean().item())
        
    ax.plot(vert_t_profile, levels, label=data_sources[i])
    
ax.set_title(plot_name)
ax.set_ylabel("p (hPa)")
ax.set_xlabel("Avg Pert Temp (degC)")
ax.yaxis.set_inverted(inverted=True)
ax.legend()
fig.savefig("stuve_comparison.png", dpi=300)

# print("PERT MAX ABS DIFFS")
# for k, v in diff_pert.items():
#     print(f"{k}: {v}")

# e2s_IC = xr.open_dataset(
#     "/N/slate/jmelms/projects/IC/JAS_2016-2020/IC_files/Pangu24.nc"
# )
# hm24_IC = xr.open_dataset("/N/slate/jmelms/projects/IC/JAS_HM24/IC_files/Pangu24.nc")

# diff_IC = {k: v.values.item() for k, v in dict(np.abs(e2s_IC - hm24_IC).max()).items()}

# print("\n\n")
# print("IC MAX ABS DIFFS")
# for k, v in diff_IC.items():
#     print(f"{k}: {v}")
