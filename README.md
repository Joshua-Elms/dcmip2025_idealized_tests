This repository contains code that will be used to support DCMIP2025; it is currently a work in progress.

It currently consists of:

 * `dcmip2025_helper_funcs.py` to assist with creating real and idealized initial conditions for the `fcnv2_sm` (SFNO) `earth2mip` model
 * `test_real_initial_condition.py` to demonstrate using these helper functions to generate initial conditions from ERA5
 * `test_isothermal_atmosphere.py` to demonstrate using these helper functions to generate an idealized initial condition. *(Note that this test results in unrealistic output from the model, which is likely due to how different it is from the model's training data.)*
 * `submit_*.qsub` to demonstrate launching these tests on NCAR's casper

### Job help

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
qsub -A UMIC0107 -q casper -N qvscode_jmelms -l select=1:ncpus=1:mem=300GB:ngpus=1 -l walltime=6:00:00 -j oe -o /glade/derecho/scratch/jmelms/.qvscode_logs/qvscode.log -v walltime_seconds=21600 /glade/u/apps/opt/qvscode/bin/launch.pbs
code --remote ssh-remote+jmelms@casper07.hpc.ucar.edu # to actually reconnect from vscode on login node
```

To check how much storage you are using, run `gladequota`

To check full job info, run `qstat -f -u jmelms`


### Ideas for Future Work
It might be useful for the DCMIP participants to have some ideas on what they could implement above-and-beyond these experiments. These ideas will encompass new experiments and modifications to existing ones. 
1. Something about moist atmospheric mass in the mass cons. experiment
2. Comparing the baseline OLR experiment ($\Delta T = 0$) to the actual forecast provided, accessible either (perhaps) through ECMWF or definitely on Glade @ `e5.oper.fc.sfc.accumu.128_179_ttr.ll025sc.2020010106_2020011606.nc` (`e5.oper.fc.sfc.accumu.128_179_ttr.ll025sc.{yyyy}{mm}{dd}{hh}_{yyyy}{mm}{dd}{hh}.nc`)
3. 