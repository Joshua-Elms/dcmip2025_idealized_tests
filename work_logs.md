### 04/15

#### Conservation of Mass

Finished this test yesterday, now I'm rerunning a longer simulation (224 tsteps / 8 weeks) at Travis' suggestion to see whether the model stabilizes somewhere around 985 hPa or keeps moving. 


#### Conservation of Energy

Finally time to start this guy. Let's write out steps as we go:

1. Build "ERA5 OLR value" downloader. It was tricky even finding OLR in ERA5, but I think I've got it: . It's an accumulation field, which is an integral over some period of time, so the first step was to figure out that interval. It's almost certainly an hour; both this [site](https://confluence.ecmwf.int/pages/viewpage.action?pageId=82870405#heading-Meanratesfluxesandaccumulations) and my testing (download global OLR at some time, compute mean, divide by 240 W/m^2 to get seconds, divide by 60 to get minutes and the result is about 56 minutes).
2. Now I need to set up the inference script for this experiment... shouldn't be hard, but I'd like to test the utils for this purpose instead of just rewriting the same loop. Not that much has changed here, after all. Well maybe that's bunk, it's a delta exp... we'll see. 
3. Finally, need to wrap it all together with an analysis script that takes in all the data produced and outputs a simple plot. 

TODO: 
- Item 3
- Improve config file; I think most of them should have a `keep_vars` field so that you can add vars if you decide you want to do more analysis, even though the default script will remove everything unnecessary. This all needs to be well-documented, too, if it's going to be shared and used by others. 

### 04/09

1. Reorganizing repository. `experiments` is one of only two top level directories, with each experiment getting a folder inside it. Anything that is (or could easily be) shared by multiple experiments can be lifted to `utils`, but then it has to stay functionally the same or risk breaking other working experiments. Within an experiment directory, there will be a few files:
  - `config.yaml` contains user-specified parameters for the experiment. Make sure to re-run the experiment (all numbered files in order) after modifying this. 
  - `gen_ic.py` will create the initial condition used in the experiment
  - `run_<model>.py` is the python script to run that particular model
  - `postprocess.py` is an optional script to modify the inference output if something complex or computationally intensive needs to happen before analysis
  - `analysis.ipynb` is the notebook where analysis can be developed, but 
  - `analysis.py` will be the eventual source of any figures produced by this experiment
  - `data/` contains soft links to the initial conditions, metadata, raw output, and postprocessed output. Performed w/ `ln -s <real_dir> <link_dir>`
  - `plots` contains the visualizations produced from the experiment output
  - `README.md` describes the experiment and how to run it

### 03/27

1. Found source of mysterious inference failure (shape mismatch) bug Garrett ran into. Turns out graphcast caches the weights in a `.cache.pkl` file in the working dir, but names the file the same thing regardless of which version of graphcast ("graphcast" or "graphcast_small") is running. The logic within graphcast is to try loading from any `.cache.pkl` it can find immediately, so if you first run "graphcast" and it caches those weights, your next run of "graphcast_small" will fail, and vice-versa. Two possible fixes: a) require distinct working dirs (and therefore cached graphs) between graphcast runs or b) modify earth2mip.networks.graphcast to cache graphs with a model-specific filename. The former is clanky imo, the latter is hacky unless I make a PR to earth2mip... maybe it's time to learn how to do so. 


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
  4. [X] Clean up and document these scripts
  5. [X] Set long simulation running
  6. [X] Calculate Total Energy
  7. [X] Convert Total Energy of heating into spherical butter mass
  8. [X] Move personal conda env to $WORK


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