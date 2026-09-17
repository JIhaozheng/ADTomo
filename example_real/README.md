# Real-catalog example

`data/events.csv`, `data/stations.csv`, and `data/picks.csv` are the supplied
real catalog and must remain unchanged. `00_gen_velocity.py` creates the
single initial Vp/Vs model required by the inversion on its fixed geographic
region; unlike the synthetic workflow, this real catalog has no true model or
pick-generation step.

After installing the package in editable mode and building the two extensions,
run:

```bash
cd example_real
python 00_gen_velocity.py
torchrun --standalone --nproc_per_node=8 05_inversion.py
```

The script deterministically selects 1,000 catalog events by default, retains
their associated stations and P/S picks, maps each absolute pick time to its
catalog event ID, and fits
`phase_time - event_time` with direct P and S travel times. Each rank owns a
disjoint station set and builds only those station-local grids.

Set `ADTOMO_EVENT_COUNT` and `ADTOMO_EVENT_SEED` to choose a different
reproducible subset. Set `ADTOMO_EVENT_COUNT=12988` to run the full catalog.

The fixed global model uses 0.05-degree horizontal and 2-km depth spacing;
station-local forward grids use 2-km spacing. To use one CPU process instead,
run `python 05_inversion.py` after the initial-model step.

By default, the real-catalog run uses perturbation-smoothness weights
`lambda_vp=lambda_vs=0.1`. Override them with `ADTOMO_LAMBDA_VP` and
`ADTOMO_LAMBDA_VS`; optional damping weights use `ADTOMO_ALPHA_VP` and
`ADTOMO_ALPHA_VS`. Use zero weights for the unregularized objective. DDP
evaluates each rank's station set in one forward calculation and synchronizes
gradients once per Adam step. Stations are greedily balanced by estimated
local forward-grid cells and phase count, while each rank weights its local
loss by pick count before gradient reduction.
By default each rank uses an equal share of the CPUs assigned to the job; set
`ADTOMO_THREADS_PER_RANK` to override that value.

Results are written under `results/`: `model_inverted.pt` and the
phase-time MSE figure `inversion_progress.png`. One untransposed
longitude/latitude velocity figure is saved for every model depth under
`results/depth_slices/`.
