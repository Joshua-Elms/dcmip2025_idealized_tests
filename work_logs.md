### 02/21 cont'd 02/23

Goal for the day: get energy conservation experiment running on casper. 

Steps: 
  1. [X] Package all changes up on BR200, push, pull down to casper
  2. [X] Parametrize python script for job submission and ICs 
    - Still working on this
    - Current status: getting gen_range_data.py up to snuff
    - Tasks within that:
      1. get lat/lon.npy files from BR200 into my metadata folder here
      2. generate levels from fortran_metadata.py
      3. make sure I/O locs correct
      4. Individual trial of gen_IC_FCN.out
      5. Run script standalone to produce data
  3. [X] Test-run small, short job submission and reading data
  4. [ ] Clean up and document these scripts
  5. [X] Set long simulation running
  6. [ ] Calculate Total Energy
  7. [ ] Convert Total Energy of heating into spherical butter mass
  8. [ ] Move personal conda env to $WORK


### 02/12

Goal for the day: calculate and plot mean time-to-convergence of initial states

Steps: 
1. make base IC (Czech)
  - use bouvier et al. 2024 IC w/ varying zt0 (doing this one) OR 
    1. get regular bouvier code running
    2. port it to dcmip helper fmt
  - download and average 30 year global climatology of FCN vars
3. generate n scenarios of global warming -30 to +30 degrees about mean (Czech)
4. run all scenarios (Czech)
5. compute global-mean temp for each model
6. plot time series

I completed all steps. 

In short: 
1. Initial conditions are first generated in the .F90 files from Bouvier
2. They are processed in the utils.py-called script in IC/bouvier
  1. One pass gets them from fortran -> weird python
  2. Another gets them from weird python -> dcmip2025 fmt
3. Once processed, they are read in by the job script as each new inference step runs
4. analysis dir has ipynb to read in data and plot lat weighted global mean sfc temp, but will need to see about moving this to moist dynamic energy or however one defines earth's conservation of energy