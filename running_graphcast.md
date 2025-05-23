## Notes (05/17)
---
1. Using env `earth2mip_graphcast` located here: `/glade/work/jmelms/software/miniconda3/envs/earth2mip_graphcast`. This has the customizations needed to run tendency reversion tests and use graphcast in the Earth2MIP framework, so I think we'll need to use it (or a similar version) for DCMIP. Unfortunately I haven't kept a greaaaat log of the modifications I've made to it, but it's a clone of e2mip and therefore git status reveals all the ways in which it's been changed. 
2. We're planning to use `graphcast_operational` for DCMIP25 because it most closely matches our other models; it's 0.25° horizontal, 13-level, 6-hour timestep model in u, v, w, T, z, q, and some single-level fields (doesn't use 6-hourly precipitation, which simplifies the IC a lot). 
3. GPU memory consumption for `graphcast_operational` is ~60GB, which means we definitely still need the H100s for it. We speculated that since it had 1/3 as many levels as `graphcast` it would be about 3x as efficient, but it does not appear to be so. 
4. Garrett said as much from the beginning, but all versions of `graphcast` take two input timesteps to produce one output timestep, and run autoregressively: $$x_{t+1} = M(x_{t-1}, x_{t})$$ This complicates the experiments a little bit. In the Hakim and Masanam (2024) steady tropical heating experiment, for example, should both $t=-1$ and $t=0$ have the heating perturbation applied, or should it be absent in the first and added only to the second? We'll have to think about how to structure the two ICs based on this. 
5. All versions of graphcast claim to use `top_net_longwave_radiation` as a forcing for the model, but they run just fine if you don't provide the model with any particular values of this forcing. What's going on here? Well, turns out that "not providing a forcing" doesn't actually do anything. If you leave the `forcing` input blank, the model calls the function `tplwr` and uses the date of the run to derive incoming radiation. 
6. I made a few changes to `inference_graphcast.py` as well, so make sure to copy that file before running. 
7. Currently, `graphcast` seems to run fine (see output `era5_ic_graphcast.nc`), but neither the homebrew or standard method of running `graphcast_operational` is producing anything usable, even after making sure all input arrays were correct. Or not? Let's see... turns out the standard code for `graphcast_operational` is apparently not finding the correct values for MSL? Let's look into why this is, because then we can get that running and have something to compare the homebrew against it. Solution: just had to undo the problem I created with my tp06 fixer in the cds.py file. Now the standard `graphcast_operational` run produces reasonable output! Time to work on getting the same from the homebrew. One possible problem: graphcast appears to want latitudes from -90 to 90, but the data comes out of ERA5 (and is used for SFNO) from 90 to -90. Once flipped, output should be correct? If you fix another silent-but-violent bug that I left in the inference utils (w_file generated differently than all other vars!), then yes, it works as expected. 
8. The code in graphcast.py (from earth2mip) and inference_graphcast.py *both* assume that you're running `graphcast_operational`. There are at least 2 or 3 shape-related things that would probably break if you tried to run `graphcast_small` or `graphcast` (full) in this pipeline, especially w.r.t. tendency reversion (implemented at earth2mip/networks/graphcast.py(428)step()). 
9. I tentatively tender the following opinion: tendency reversion (TR) isn't going to be useful for non-steady state initializations, since graphcast is a binary function (two input timesteps). Take t=0 as our initialization time, which means that we also use t=-1 for the first inference timestep. Drawing from ERA5, $x_{t=0} \neq x_{t=-1}$. Now take tendency $$dx = M(x_{t=-1}, x_{t=0}) - x_{t=0}$$ Therefore when we perform inference with TR, we'll get $$x_{t=1} = M(x_{t=-1}, x_{t=0}) - dx = x_{t=0}$$ The following inference step will be $$x_{t=2} = M(x_{t=0}, x_{t=1}) - dx = M(x_{t=0}, x_{t=0}) - dx = M(x_{t=0}, x_{t=0}) - M(x_{t=-1}, x_{t=0}) + x_{t=0}$$ Instead of the clean "back to the initial condition" inference we expect, we begin to heap tomfoolery our beautiful initial condition ad nauseum. In other words: steady-state $x$ ($\overline{x}$) is required to ensure that the initial (t=0) and pre-initial (t=-1) timesteps of $x$ have the same state, such that the tendencies calculated for them don't cause drift in the model over time. The implication of this is that we can still use TR for the H&M24 experiments and baroclinic wave tests, but we should avoid it during mass/energy conservation experiments, individual-ERA5-date case studies, and other non-steady-state applications. This only holds for graphcast -- it's still doable for other experiments, but the underlying assumption of steady-state $x$ that's mentioned in H&M24 is perhaps wise to think more about. 

## Experiment Descriptions
---
`experiments` is one of only two top level directories, with each experiment getting a folder inside it. Anything that is (or could easily be) shared by multiple experiments can be lifted to `utils`, but then it has to stay functionally the same or risk breaking other working experiments. Within an experiment directory, there will be a few files:
  - `config.yaml` contains user-specified parameters for the experiment. Make sure to re-run the experiment (all numbered files in order) after modifying this. 
  - `run_<model>.py` is the python script to run that particular model
  - `analysis.ipynb` is the notebook where analysis can be developed, but 
  - `analysis.py` will be the eventual source of any figures produced by this experiment
  - `data/` contains soft links to the initial conditions, metadata, raw output, and postprocessed output. Performed w/ `ln -s <real_dir> <link_dir>`
  - `plots` contains the visualizations produced from the experiment output
  - `README.md` describes the experiment and how to run it

#### `bouvier_baroclinic_wave`
Structurally, the file `0.config.yaml` controls the simulation. When you are satisfied with the config and run `1.run_sfno.py`, that file will copy the config into the aforementioned `data` directory under a newly-generated folder named whatever your config's `experiment_name` parameter is set to. All ICs, output data, and plots produced by this run of the experiment will be in this folder, which helps to keep things organized and ensure you know the configuration used for that experiment. `2.analsyis.py` just produces plots, but I don't think we need to worry as much about plotting for now, since our primary concern is getting DCMIPers data to analyze themselves. 

The details of the experiment *ought* to be laid out in `bouvier_baroclinic_wave/README.md`, but I may or may not have done that yet. In short, we use the fortran IC generator from Bouvier et al. (2024) to produce a steady-state, hydrostatic- and geostrophic-ally balanced initial condition for inference. We can apply tendency reversion to great effect here (the plots I sent you, Garrett) to show how a small perturbation to the geopotential field produces a baroclinic wave, but also allows instabilities into the model which culminate in a pulsating planetary wave of uncertain origin (in SFNO, anyways... we'll see what graphcast does!)


#### `hakim_and_masanam`
The structure here is not quite as clean as in the other one. While the config setup is the same, the run scripts don't do the whole "create a named sub-dir for this particular version of the experiment's output to go into". Rather, all data goes into the `data` directory and all plots into the `plot` directory, which means the onus is on the user to name files such that they don't overwrite. I'd like to restructure this experiment (and all the others) to use the same flow as in `bouvier_baroclinic_wave`, but that's not super urgent. I'll probably get to it in the week before DCMIP. 

## Remaining Work
---

#### Experiments
1. `bouvier_baroclinic_wave`
  1. Change output vars to work for `graphcast_operational`, not just SFNO. TODO items listed in this file: `experiments/bouvier_baroclinic_wave/initial_conditions/utils.py`
  2. Add `model` field to config to allow inference with any. Should get passed on to IC generator to work w/ step 1. 
  3. Flesh out `README.md`. Current version is outdated (just a description of total energy), but it should actually include some literature review, background about the experiment, details on how to run it, and perhaps even hints for the analysis. In short, I want it to eventually overlap with the [DCMIP wiki page](https://sites.google.com/umich.edu/dcmip-2025/) for this experiment. 
2. `hakim_and_masanam`
  1. Calculate DJF & JAS means using updated scripts in supplementary. `compute_mean` might need some work, but I think the downloader is solid (maybe check whether specific humidity is really called `specific_humidity` in the CDS API). 
  2. Update config and inference scripts to work for graphcast as well as SFNO. 
  3. Update README with more background, see point 1.3 above.
  4. Handle the config and experiment flow as in `bouvier_baroclinic_wave`, see H&M24 experiment description above. 
3. `mass_conservation`
  1. Update config to choose model
  2. Update inference script to accept diff models
  3. Update README to include theory underlying this exper
  4. Handle the config and experiment flow as in `bouvier_baroclinic_wave`
4. `one_step_energy_conservation`
  1. Update config to choose model
  2. Update inference script to accept diff models
  3. Update analysis to calculate total energy differently depending on model, i.e. graphcast uses q instead of r but doesn't have TCWV
  4. Handle the config and experiment flow as in `bouvier_baroclinic_wave`

#### Utils
---
1. There's some better organizational system to be found here... perhaps consult Travis on what it should be. Diff utils file per model? Mono-file and parametrize by model? Unsure, and it's midnight so I'm *really* unsure. This one can come later. 
2. Run chatgpt over it in bug-checker-mode to see whether anything obvious is screwed up


#### Infra
---
1. Make a copy of my env `earth2mip_graphcast`. It's got all the code necessary both to run graphcast and to handle tendency reversion for SFNO/graphcast_operational, so we'll need it for the participants. Some sort of exact-copy function would be ideal here; everybody gets a version of it in case they want to mess around, but it'll have all the same dylibs and etc. so that nobody is stuck in a dependency-resolution nightmare during the workshop. 