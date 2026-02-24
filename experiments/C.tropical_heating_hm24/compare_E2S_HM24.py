import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

e2s_heating = np.load("heating_E2S.npy")
hm24_heating = np.load("heating_HM24.npy")

diff_heating = e2s_heating - hm24_heating

e2s_IC = xr.open_dataset("/N/slate/jmelms/projects/IC/DJF_1979-2019/IC_files/Pangu24.nc")
hm24_IC = xr.open_dataset("/N/slate/jmelms/projects/.E2S_cache/pangu_hm24_test/mean_DJF.h5")

diff_IC = e2s_IC - hm24_IC

breakpoint()
