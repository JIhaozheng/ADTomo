# ADTomo

ADTomo is a small CPU implementation of differentiable spherical eikonal
tomography. It has exactly two grids:

1. A global velocity model `V(depth, latitude, longitude)`.
2. A local Cartesian East/North/Down fast-sweeping grid for each station.

The Earth remains spherical. Geographic points are converted to ECEF, rigidly
translated and rotated into the station-local END frame, then sampled from the
global model with PyTorch interpolation.

## Conventions

- longitude: degrees east; latitude: degrees north; depth: km positive down
- velocity: km/s; time: seconds; Earth radius: 6371 km
- global model tensor order: `(depth, latitude, longitude)`
- local forward-field tensor order: `(z_local, y_local, x_local)` = `(Down, North, East)`
- local physical-coordinate order: `(x_local, y_local, z_local)` = `(East, North, Down)`
- retained C++ kernels: CPU-only, `torch.float64`, and one isotropic spacing

The C++ kernels store local fields in physical `(x_local, y_local, z_local)`
order. The Python wrappers hide that detail and expose local arrays in
`(z_local, y_local, x_local)` order. This convention never applies to the
global spherical model.

## Catalog times

The catalog preserves ISO origin and phase timestamps. Every pick is converted
to its event-relative observed time:

```text
phase_dt = phase_time - catalog_event_time
```

The modeled time is:

```text
predicted_phase_dt = event_dt + travel_time
```

`event_dt` has one value per catalog event, not per pick. For velocity
inversion it is fixed to zero. A future joint inversion can make it trainable;
then `new_event_time = catalog_event_time + event_dt`.

## Install

```bash
cd /path/to/ADTomo_hz
pip install -r requirement.txt
python setup.py build_ext --inplace
pip install -e . --no-build-isolation
```

The first build downloads Eigen and compiles only `eikonal2d_op` and
`eikonal3d_op`.

To run one test script from the test directory itself, build the extensions
once, then use:

```bash
cd /path/to/ADTomo_hz/tests
python test_eikonal.py
python test_coordinate.py
python test_grid.py
python test_gradient.py
```

Each test file is directly executable and adds the adjacent source checkout to
Python's import path.

## Synthetic workflow

```bash
python examples/00_gen_velocity.py
python examples/01_gen_stations.py
python examples/02_gen_events.py
python examples/03_gen_picks.py
python examples/04_forward.py
python examples/05_inversion.py
```

Catalog files are written to `examples/data/`; the inversion writes one final
model and `inversion_progress.png` to `examples/results/`.

The complete forward path stays visible:

```python
grid = ForwardGrid(station_lonlatdepth, event_lonlatdepth, model, spacing=5.0)
predicted_phase_dt = predict_phase_times(model, grid, "P", event_dt)
loss = ((predicted_phase_dt - observed_phase_dt) ** 2).mean()
loss.backward()
```
