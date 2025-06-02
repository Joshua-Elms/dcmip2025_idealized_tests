# Hakim & Masanam (2024) Experiments

This experiment runs SFNO, Graphcast, or Pangu following the Hakim and Masanam protocol in which the model tendencies are constrained such that an unperturbed version of the model runs in steady state.  The perturbed versions aim to isolate the response of the model to various perturbations: e.g., tropical heating.  This experiment aims to satisfy both objectives (2) and (3): *run idealized simulations with these two models to probe aspects of the models' physical fidelity; and explore and intercompare model responses as the model inputs stray further and further from their training dataset*.

## Instructions

The scripts/notebooks in this folder are labeled and arranged in the order in which they should be modified and/or run.

### `0.config.yaml`
The `0.config.yaml` file contains variables that can be configured to modify the behavior of the simulation.

### `1.run_sfno.py`

TODO

### `2.analysis.py`

TODO

### Experiment Selection

Use the `hm24_experiment` parameter in the config file to switch between the experiments. The four options are: 
1. `tropical_heating`
2. `extratropical_cyclone`
3. `geostrophic_adjustment`
4. `tropical_cyclone`

### Experiment-Specific Config Options
Check general config information at [`experiments/README.md`](../README.md) for options shared by multiple experiments. 

Parameters shared by all HM24 experiments:

- `time_mean_IC_dir`: Path to the directory that contains model-specific time-mean initial conditions drawn from ERA5. Default is `/glade/derecho/scratch/jmelms/dcmip/era5_time_means`, but you can create your own using the `CDS_downloader.py` and `compute_ERA5_time_mean.py` scripts under `experiments/B.hakim_and_masanam/supplementary`. 
- `IC_season`: Either `DJF` (December, January, and February) or `JAS` (July, August, and September); used to select IC file from above directory. HM24 experiments 1-3 are typically run with `DJF` and experiment 4 is run with `JAS`. 
- `hm24_experiment`: See section on [experiment selection](#experiment-selection). 

(1) Tropical Heating Parameters

- `perturbation_params`: See [HM24 methods](https://journals.ametsoc.org/view/journals/aies/3/3/AIES-D-23-0090.1.xml#d2752741e274) for more details on these parameters. 
    - `amp`: Amplitude of perturbation
    - `k`: horizontal wavenumber of perturbation cosine wave. Perturbation longitudinal extent defined as lon where wave < 0.
    - `locRadkm`: Size of perturbation in km, only controls latitudinal extent of pert. 
    - `ylat`: Latitude (-90 to 90 °N) of center of perturbation
    - `xlon`: Longitude (0 to 359.75 °E) of center of perturbation

(2) Extratropical Cyclone (ETC) Parameters:
- `perturbation_params`: 
    - `amp`: Scale factor for HM24 ETC perturbation. `0.5` will halve the original perturbation, for example. 

(3) Geostrophic Adjustment (GA) Parameters:
- `perturbation_params`: 
    - `amp`: Scale factor for HM24 GA perturbation. `0.5` will halve the original perturbation, for example. 

(4) Geostrophic Adjustment (TC) Parameters:
- `perturbation_params`: 
    - `amp`: Scale factor for HM24 TC perturbation. `0.5` will halve the original perturbation, for example. 

## Details

This is an implementation of the four experiments provided in Hakim & Masanam's 2024 paper on machine learning forecast model testing, henceforth "HM24". 

These four experiments are: 
1. Steady tropical heating
2. Idealized extratropical cyclone (ETC) development 
3. Geostrophic adjustment
4. Tropical cyclone (TC) formation

### Initial Conditions
---
Experiments 1-3 take as their initial condition (IC) the ERA5 December-February (DJF) 0z 1979-2019 time mean. 

Experiment 4 is the same, but for July-September (JAS).

These climatological means are treated as a steady-state, and are therefore notated as $\overline{x}_\text{w}$ and $\overline{x}_\text{s}$ for winter ($\text{w}$, DJF) and summer ($\text{s}$, JAS), respectively. 

The scripts used to download the necessary ERA5 climatology are located in the `supplementary` folder. Due to the size of the data that goes into each IC (~400 GB), the download scripts are built to run in parallel. Running them with at least a few parallel workers yields significant improvements. All data is sampled at 10 day intervals, using the following dates in each month: 
- December: 01, 11, 21, 31
- January: 10, 20, 30
- February: 09, 19
- July: 01, 11, 21, 31
- August: 10, 20, 30
- September: 09, 19, 29

### GC Localization Function

### Idealized Cyclone Perturbations

### Experiment

### References
1. Hakim, G. J., and S. Masanam, 2024: Dynamical Tests of a Deep Learning Weather Prediction Model. Artif. Intell. Earth Syst., 3, e230090, https://doi.org/10.1175/AIES-D-23-0090.1.
2. GC paper...
3. ERA5 data
4. SFNO
5. GraphCast
6. PanGu