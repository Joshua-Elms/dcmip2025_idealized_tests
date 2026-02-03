import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

g = 9.81 # m/s^2

hm24_IC_path = Path("/N/slate/jmelms/projects/.E2S_cache/pangu_hm24/mean_DJF.h5")
hm24_IC_ds = xr.open_dataset(hm24_IC_path)
a = hm24_IC_ds
a_z500 = a["mean_pl"].values[0, 5] / g

plt.imshow(a_z500)
plt.colorbar()
plt.savefig("a_z500.png")
plt.close()
plt.clf()

e2s_IC_path = Path("/N/slate/jmelms/projects/IC/DJF_1979-2019/IC_files/Pangu24.nc")
e2s_IC_ds = xr.open_dataset(e2s_IC_path)
b = e2s_IC_ds
b_z500 = b["z500"].squeeze().values / g

plt.imshow(b_z500)
plt.colorbar()
plt.savefig("b_z500.png")
plt.close()
plt.clf()

diff = b_z500 - a_z500

plt.imshow(diff)
plt.colorbar()
plt.savefig("diff.png")
plt.close()
plt.clf()
