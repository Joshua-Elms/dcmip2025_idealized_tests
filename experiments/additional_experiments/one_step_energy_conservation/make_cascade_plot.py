import xarray as xr
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
import datetime as dt
from utils import inference_sfno, vis
import scipy
from time import perf_counter
import torch
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm

ds = xr.open_dataset("data/processed_output_2018_each_month.nc")

# Extract the data for the first initial condition (ic_date = 0)
ds_ic0 = ds.isel(init_time=0)

# Get unique delta_T values
delta_Ts = ds_ic0.delta_T.values

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Choose a colormap that distinguishes the lines well
cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, len(delta_Ts)))

lead_times_h = np.arange(0, 6*ds_ic0.lead_time.size, 6)

# Plot a line for each delta_T
for i, delta_T in enumerate(delta_Ts):
    ds_subset = ds_ic0.sel(delta_T=delta_T)
    time_hours = lead_times_h  # Assuming 6-hour timesteps
    
    # Plot the total energy time series
    ax.plot(time_hours, ds_subset.LW_TE.values.squeeze(), 
            color=colors[i], 
            linewidth=2,
            label=f'ΔT = {delta_T:.1f} K')

# Customize the plot
ax.set_xticks(time_hours)
ax.set_xlabel('Time (hours)', fontsize=12)
ax.set_ylabel('Total Energy (J/m²)', fontsize=12)
ax.set_title('Total Energy Evolution for Different Temperature Perturbations', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(title='Temperature Perturbation', fontsize=10)

# Scientific notation for y-axis if values are large
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Save the figure
plt.tight_layout()
plt.savefig('plots/total_energy_evolution.png', dpi=300, bbox_inches='tight')

plt.show()