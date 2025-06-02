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

**A. Mass Conservation:** This experiment runs SFNO, Graphcast, or Pangu in standard re-forecast mode: taking initial conditions from ERA5 and running the model forward a set amount of time. This experiment aims to satisfy objective (1): *gain familiarity with running three ML weather forecast emulators.*

**B. Hakim and Masanam:** This experiment runs either SFNO, Graphcast, or Pangu following the Hakim and Masanam protocol in which the model tendencies are constrained such that an unperturbed version of the model runs in steady state.  The perturbed versions aim to isolate the response of the model to various perturbations: e.g., tropical heating.  This experiment aims to satisfy both objectives (2) and (3): *run idealized simulations with these three models to probe aspects of the models' physical fidelity; and explore and intercompare model responses as the model inputs stray further and further from their training dataset*.

**C. Bouvier Baroclinic Wave:** This experiment runs either SFNO, Graphcast, or Pangu using an idealized, zonally-symmetric (unless perturbed) aquaplanet initial condition.  The initial condition is designed to be baroclinically unstable, such that perturbing the initial condition should spontaneously result in the growth of a baroclinic wave. The initial condition protocol for this experiment was designed by Clement Bouvier: a DCMIP 2025 attendee! This experiment aims to satisfy both objectives (2) and (3): *run idealized simulations with these three models to probe aspects of the models' physical fidelity; and explore and intercompare model responses as the model inputs stray further and further from their training dataset*.

More details about each experiment can be found in the `experiments/*/README.md` files.

### General Config Options
- `n_timesteps`: Number of 6-hour steps the model will move forward during inference. Output netcdf "lead_time" dimension will be one longer than this variable because the initial condition ($t=0$) is saved. For example, `n_timesteps: 12` simulates 3 days and produces 13 total timesteps in the output dataset.
- `device`: Desired hardware to drive inference. "cuda" runs job on GPU, which is faster by 1-2 orders of magnitude than "cpu" and therefore preferred. DCMIP25 should have 2 L40 GPUs available explicitly for the ML group, but if those are busy or down, running on CPU should work (slowly). 
- `experiment_dir`: Path to a dedicated directory, high-capacity directory for experiment output. The `setup.py` script in this repository should have created one such directory per experiment, formatted for the config as `/glade/u/home/YOURUSERNAME/DCMIP2025-ML/experiments/A.mass_conservation/data` (no quotes, modify path as needed). It is recommended that this path be a symlink, with the real directory stored on your $SCRATCH or $WORK directory, and the pointer directory under the experiment, e.g. `.../A.mass_conservation/symlink_dir`. 
- `experiment_name`: The particular name for this iteration of the experiment. For example, you might want to run experiment (A) Mass Conservation for one simulated year (n_timesteps = 1460), and so you could provide the experiment name as `one_simulated_year`. A folder with the same name will automatically be created underneath the `experiment_dir`, and all output (data, initial conditions, plots, etc.) from this iteration of the experiment will be placed there.
- `tendency_reversion`: `true` or `false` to activate or deactivate tendency reversion (TR) feature discussed below. 

### Tendency Reversion

For some atmospheric state vector $x$ at $t=0$ ($x_0$), the models ($M$) in this study will act on $x_0$ to move it forward by one timestep: $M(x_0) = x_1$. The change in $x$ from $t=0$ to $t=1$ is called the "tendency", and is notated as $dx = x_1 - x_0$. 

If we do not perturb the initial condition at all, we can ensure the model produces the same output at each timestep by subtracting (or "reversing") this tendency each time. Output which is produced with tendency reversion (henceforth TR) enabled will be flagged with a prime: $x_1' = M(x_0) - dx$. In this example, $x_1' = x_0$. Try verifying this algebraically using the information above if it's not intuitive. 

Running the model with TR enabled and an unperturbed initial condition is a good test of the TR mechanism implemented in the code, but it's not scientifically interesting to produce a sequence of identical vectors $x_n' = \ldots = x_1' = x_0$. 

Instead, consider modeling the initial state $x_0$ with the addition of some slight perturbation $p$: $x_1' = M(x_0 + p) - dx$. If we assume that the model function $M$ is approximately linear w.r.t. the model state, then we can decompose the above into 
$$
x_1' = M(x_0) + M(p) - dx = x_1 - dx + M(p)
$$ 

We then use substitute the definition of our tendency $dx = x_1 - x_0$ to find 
$$
x_1' = x_0 + M(p)
$$
From here, we simply subtract away the initial state vector $x_0$ to isolate the model response to the perturbation, $M(p)$. Tendency reversion (TR) and the assumption of approximate model linearity let us see how a perturbation ripples through the model solution early on. This is not indefinitely useful, though; the assumption of linearity w.r.t. the initial perturbation fails as we pass a few simulated weeks, in our experience. 