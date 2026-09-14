# Real-catalog example

`events.csv`, `stations.csv`, and `picks.csv` are the supplied real catalog.
They are read-only inputs; the numbered synthetic-generation scripts in this
directory are not part of this real-data workflow.

After installing the package in editable mode and building the two extensions,
run:

```bash
cd example_real
torchrun --standalone --nproc_per_node=8 05_inversion.py
```

The script uses all 12,988 catalog events, 426 stations, and 925,982 P/S
picks. It maps each absolute pick time to its catalog event ID and fits
`phase_time - event_time` with direct P and S travel times. Each rank owns a
disjoint station set and builds only those station-local grids.

The full-domain model and local forward grids use a deliberately coarse
0.1-degree/5-km spacing for this first full-catalog result. To use one CPU
process instead, run `python 05_inversion.py`.

By default, the real-catalog run uses perturbation-smoothness weights
`lambda_vp=lambda_vs=0.1`. Override either value through
`ADTOMO_LAMBDA_VP` or `ADTOMO_LAMBDA_VS`; use zero for the unregularized
objective. DDP evaluates each rank's station set in one forward calculation
and synchronizes gradients once per Adam step.

Results from the original-style rank-parallel workflow are written under
`results/`: `model_inverted_old_ddp.pt` and the untransposed
longitude/latitude diagnostic figure `inversion_progress_old_ddp.png`.
