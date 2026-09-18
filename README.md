# ADTomo

A differentiable travel-time tomography framework based on the eikonal equation.

## Install

```bash
pip install -r requirement.txt
python setup.py build_ext --inplace
pip install -e . --no-build-isolation
```

## Synthetic pipeline

The same four steps generate a 3-D checkerboard dataset (`--model 3d`, the default)
or a 1-D depth-profile dataset (`--model 1d`); the picks step follows whichever
model is in `data/model_true.pt` (`ForwardGrid` + 3-D eikonal, or `ForwardGrid2D`
+ 2-D eikonal on the station-centered spherical section):

```bash
python examples/00_gen_velocity.py --model 3d          # or --model 1d
python examples/01_gen_stations.py --num-stations 50
python examples/02_gen_events.py --num-events 500 --location-noise-km 2 --time-noise-s 0.5
python examples/03_gen_picks.py
```

`02_gen_events.py` writes the true catalog `data/events.csv` and the catalog the
inversion starts from, `data/events_initial.csv` (identical unless noise is given).
The model generator accepts `--wavelength-lon/--wavelength-lat/--wavelength-depth`
and `--amplitude`.

## Inversion

`inversion3d.py` and `inversion1d.py` share one workflow: choose which parameters are
trainable (`vp`, `vs`, `event_loc`, `event_time`, toggled through `requires_grad` as
in AI4EPS/ADTomo), then minimize the arrival-time misfit with L-BFGS (default) or Adam.

```bash
MODEL=3d bash examples/run_inversion.sh                                   # velocity only
MODEL=1d TRAINABLE=event_loc,event_time bash examples/run_inversion.sh    # relocation only
MODEL=1d TRAINABLE=vp,vs,event_loc,event_time bash examples/run_inversion.sh   # joint
NPROC=4 ITERATIONS=50 OPTIMIZER=adam LEARNING_RATE=0.01 MODEL=3d bash examples/run_inversion.sh
```

Other knobs: `SPACING`, `LAMBDA_VP/VS` (smoothness), `ALPHA_VP/VS` (damping); any extra
arguments are passed through to the script (e.g. `--grid-padding 15` for the 1-D section,
whose fixed forward grids must contain the events throughout the relocation). `run_pipeline.sh` chains all steps:

```bash
MODEL=1d LOCATION_NOISE_KM=2 TIME_NOISE_S=0.5 TRAINABLE=event_loc,event_time bash examples/run_pipeline.sh
```

Results go to `examples/results/` (`model_inverted.pt`, `events_inverted.csv`) and
figures to `examples/figures/` (`inversion.png`, `events.png`).
