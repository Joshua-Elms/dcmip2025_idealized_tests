# DCMIP 2025 Machine Learning Hands-On Session

**Objectives:** Participants engaging in the DCMIP 2025 ML hands-on session will:

1. gain familiarity with running two ML weather forecast emulators: [NVidia's SFNO](https://doi.org/10.48550/arXiv.2306.03838) and [Google's GraphCast](https://doi.org/10.48550/arXiv.2212.12794);
2. run idealized simulations with these two models to probe aspects of the models' physical fidelity; and
3. explore and intercompare model responses as the model inputs stray further and further from their training dataset


## Prerequisites

To meet the above objectives in the five half-day hands-on sessions, participants will need background in the following:

* proficiency in unix command line environments
* proficiency with python and with weather/climate data visualization
* basic familiarity with machine learning
* access to NCAR supercomputing systems: specifically casper

## Getting Started

### 1. Clone this repository
e.g., `git clone https://github.com/taobrienlbl/DCMIP2025-ML.git` in your home directory

The repository contains code that will facilitate running the hands-on session experiments.  It is organized as follows:

```
experiments/ (folders containing experiment code & instructions)
    A.mass_conservation/        (the first experiment that everyone will run)
    B.hakim_and_masanam/        (one of two experiments that participants will run)
    C.bouvier_baroclinic_wave/  (one of two experiments that participants will run)
    additional_experiments/     (this won't likely be used in DCMIP 2025)
utils/  (a folder with helper code that you won't likely need to directly use)
```

### 2. Follow README instructions

Each of the above folders in `experiments/` has a `README.md` file with details about the experiments and instructions on getting started.  You can either view the README file in a text editor or you can view a nicely-rendered version of it on [the DCMIP 2025 ML experiment github site](https://github.com/taobrienlbl/DCMIP2025-ML).

## Basic details about experiments

**A. Mass Conservation:** This experiment runs either SFNO or GraphCast in standard re-forecast mode: taking initial conditions from ERA5 and running the model forward a set amount of time. This experiment aims to satisfy objective (1): *gain familiarity with running two ML weather forecast emulators.*

**B. Hakim and Masanam:** This experiment runs either SFNO or GraphCast following the Hakim and Masanam protocol in which the model tendencies are constrained such that an unperturbed version of the model runs in steady state.  The perturbed versions aim to isolate the response of the model to various perturbations: e.g., tropical heating.  This experiment aims to satisfy both objectives (2) and (3): *run idealized simulations with these two models to probe aspects of the models' physical fidelity; and explore and intercompare model responses as the model inputs stray further and further from their training dataset*.

**C. Bouvier Baroclinic Wave:** This experiment runs either SFNO or GraphCast using an idealized, zonally-symmetric (unless perturbed) aquaplanet initial condition.  The initial condition is designed to be baroclinically unstable, such that perturbing the initial condition should spontaneously result in the growth of a baroclinic wave. The initial condition protocol for this experiment was designed by Clement Bouvier: a DCMIP 2025 attendee! This experiment aims to satisfy both objectives (2) and (3): *run idealized simulations with these two models to probe aspects of the models' physical fidelity; and explore and intercompare model responses as the model inputs stray further and further from their training dataset*.

More details about each experiment can be found in the `experiments/*/README.md` files.

## Tentative schedule

* **Day 1**: Run standard forecasts and evaluate mass conservation (experiment A)
* **Day 2**: Choose one of two idealized tests (experiments B or C); run the tests and visualize results
* **Day 3**: Vary idealized test parameters & stress-test models; run with other model; visualize results
* **Day 4**: Choose another of the two idealized tests (experiments B or C); vary parameters, vary models; visualize results
* **Day 5**: Finalize & continue experiments and visualizations; prepare final presentation

As time allows, we will also work with participants to create and document entirely new idealized tests.


## Frequently used commands

There are some commands that we will use frequently in these experiments, so we provide examples below that you can copy and paste.

To run a 6 hour (default) interactive job on casper w/ 1 GPU and X GB of memory:

    qinteractive @casper -A UMIC0107 -l select=1:ncpus=1:ngpus=1:mem=300GB

To start a jupyter notebook:

    jupyter notebook --NotebookApp.allow_origin='*' --NotebookApp.ip='0.0.0.0' --port 9999

To try (unsuccessfully) to reattach to a running job, use:

    pbs_attach <job> # and once this fails
    qdel <job> # and then see line 2
    
To run a vscode session on a dedicated compute node, follow these steps: https://nhug.readthedocs.io/en/latest/blog/launch-vscode-on-casper-compute-nodes/#motivation
```
module load ncarenv/24.12
qvscode
```

Or circumvent with direct request: 
```
qsub -A UMIC0107 -q casper -N qvscode_<YOURUSERNAME> -l select=1:ncpus=1:mem=300GB:ngpus=1 -l walltime=6:00:00 -j oe -o /glade/derecho/scratch/<YOURUSERNAME>/.qvscode_logs/qvscode.log -v walltime_seconds=21600 /glade/u/apps/opt/qvscode/bin/launch.pbs
code --remote ssh-remote+<YOURUSERNAME>@casper07.hpc.ucar.edu # to actually reconnect from vscode on login node
```

To check how much storage you are using, run `gladequota`

To check full job info, run `qstat -f -u <YOURUSERNAME>`
