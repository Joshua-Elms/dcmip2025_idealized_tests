# Hakim & Masanam (2024) Experiments

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
7. ACE2