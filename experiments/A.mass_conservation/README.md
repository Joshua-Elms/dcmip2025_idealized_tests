# Mass Conservation Testing

This experiment runs either SFNO or GraphCast in standard re-forecast mode: taking initial conditions from ERA5 and running the model forward a set amount of time. This experiment aims to satisfy objective (1): *gain familiarity with running two ML weather forecast emulators.*

## Instructions

The scripts/notebooks in this folder are labeled and arranged in the order in which they should be modified and/or run.

### `0.config.yaml`
The `0.config.yaml` file contains variables that can be configured to modify the behavior of the simulation.  In the first experiment, modify the `experiment_dir` variable to point to your own directory (e.g., replace `YOURUSERNAME`).  In subsequent experiments, play around with other variables like `ic_dates` and `keep_vars`.

### `1.prep_output_dir.py`
Run this script to prepare the experiment directory that you specified in the last step.

### `2-3.submit_mass_conservation_test.pbs`
Submit a PBS job that runs steps 2 and 3 (running SFNO, GraphCast, and analyzing the results): `qsub 2-3.submit_mass_conservation_test.pbs`

Note that you can run and customize step 3 in jupyter notebook form: `3aux.analysis.ipynb`

## Variable Naming Convention for `0.config.yaml`

TODO