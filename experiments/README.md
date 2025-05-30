The experiments are organized as follows:

```
experiments\ (folders containing experiment code & instructions)
    A.mass_conservation\        (the first experiment that everyone will run)
    B.hakim_and_masanam\        (one of two experiments that participants will run)
    C.bouvier_baroclinic_wave\  (one of two experiments that participants will run)
    additional_experiments\     (this won't likely be used in DCMIP 2025)
utils\  (a folder with helper code that you won't likely need to directly use)
```

Each of the above folders in `experiments\` has a `README.md` file with details about the experiments and instructions on getting started.  You can either view the README file in a text editor or you can view a nicely-rendered version of it on [the DCMIP 2025 ML experiment github site](https://github.com/taobrienlbl/DCMIP2025-ML).

**A. Mass Conservation:** This experiment runs either SFNO or GraphCast in standard re-forecast mode: taking initial conditions from ERA5 and running the model forward a set amount of time. This experiment aims to satisfy objective (1): *gain familiarity with running two ML weather forecast emulators.*

**B. Hakim and Masanam:** This experiment runs either SFNO or GraphCast following the Hakim and Masanam protocol in which the model tendencies are constrained such that an unperturbed version of the model runs in steady state.  The perturbed versions aim to isolate the response of the model to various perturbations: e.g., tropical heating.  This experiment aims to satisfy both objectives (2) and (3): *run idealized simulations with these two models to probe aspects of the models' physical fidelity; and explore and intercompare model responses as the model inputs stray further and further from their training dataset*.

**C. Bouvier Baroclinic Wave:** This experiment runs either SFNO or GraphCast using an idealized, zonally-symmetric (unless perturbed) aquaplanet initial condition.  The initial condition is designed to be baroclinically unstable, such that perturbing the initial condition should spontaneously result in the growth of a baroclinic wave. The initial condition protocol for this experiment was designed by Clement Bouvier: a DCMIP 2025 attendee! This experiment aims to satisfy both objectives (2) and (3): *run idealized simulations with these two models to probe aspects of the models' physical fidelity; and explore and intercompare model responses as the model inputs stray further and further from their training dataset*.

More details about each experiment can be found in the `experiments/*/README.md` files.