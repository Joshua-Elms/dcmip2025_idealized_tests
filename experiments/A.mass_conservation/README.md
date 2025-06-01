# Mass Conservation Testing

This experiment runs either SFNO, Graphcast, or Pangu in standard re-forecast mode: taking initial conditions from ERA5 and running the model forward a set amount of time. This experiment aims to satisfy objective (1): *gain familiarity with running three ML weather forecast emulators.*

## Instructions

The scripts/notebooks in this folder are labeled and arranged in the order in which they should be modified and/or run.

### `0.config.yaml`
The `0.config.yaml` file contains variables that can be configured to modify the behavior of the simulation.  In the first experiment, modify the `experiment_dir` variable to point to your own directory (e.g., replace `YOURUSERNAME`).  In subsequent experiments, play around with other variables like `ic_dates` and `keep_vars`.

### `1.prep_output_dir.py`
Run this script to prepare the experiment directory that you specified in the last step.

### `2a-c.submit_mass_conservation_test.pbs`
Submit a PBS job that runs steps 2a, 2b, and 2c (running SFNO,Graphcast, and Pangu): `qsub 2a-c.submit_mass_conservation_test.pbs`

### `3.analysis.py`

Run this script (or the jupyter notebook version of it, `3aux.analysis.ipynb`) to analyze the results. 

The dependencies for this script are contained in the `analysis` conda environment, which can be set via `conda activate /glade/work/jmelms/software/miniconda3/envs/analysis`. 

### Experiment-Specific Config Options
Check general config information at [`experiments/README.md`](../README.md) for options shared by multiple experiments. 

- `ic_dates`: The dates and times for which inference will be performed. These are pulled directly from the ERA5 stored in NCAR's RDA ([linked here](https://rda.ucar.edu/datasets/d633000/)). Must be between 1940 and present day, and formatted as a list with datetime strings, e.g. `["2018/01/01 12:00", "2018/07/01 12:00"]`.
- `keep_vars`: By default, `2a.run_sfno.py`, `2b.run_gc_small.py`, and `2c.run_pangu.py` only output surface pressure (sfno) or mean sea-level pressure (sfno, graphcast, and pangu), depending on the model. If you wish to keep other fields, you can enter the field names here. SFNO options are ["Z", "R", "U", "V", "T", "VAR_10U", "VAR_10V", "VAR_100U", "VAR_100V", "VAR_2T", "TCWV"]. Graphcast_small options are ["Z", "Q", "U", "V", "T", "W", "VAR_10U", "VAR_10V","VAR_2T", "TP06"]. Pangu options are ["Z", "Q", "T", "U", "V", "VAR_10U", "VAR_10V","VAR_2T"]. Example: To add geopotential at all levels, 2m temperatures, and 100m winds (from SFNO only, Graphcast/Pangu don't have them) as output variable, use `["Z", "VAR_2T", "VAR_100U", "VAR_100V"]`.