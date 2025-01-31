This repository contains code that will be used to support DCMIP2025; it is currently a work in progress.

It currently consists of:

 * `dcmip2025_helper_funcs.py` to assist with creating real and idealized initial conditions for the `fcnv2_sm` (SFNO) `earth2mip` model
 * `test_real_initial_condition.py` to demonstrate using these helper functions to generate initial conditions from ERA5
 * `test_isothermal_atmosphere.py` to demonstrate using these helper functions to generate an idealized initial condition. *(Note that this test results in unrealistic output from the model, which is likely due to how different it is from the model's training data.)*
 * `submit_*.qsub` to demonstrate launching these tests on NCAR's casper