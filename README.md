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
- Python fields: `(z, y, x)`; physical coordinates: `(x, y, z)`
- local axes: East, North, Down
- retained C++ kernels: CPU-only, `torch.float64`, and one isotropic spacing

The C++ kernels store fields in `(x, y, z)`. The Python wrappers hide that
detail, so all public fields use `(z, y, x)`.

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

For velocity inversion, `event_dt` is fixed to zero. A future joint inversion
can make it trainable; then `new_event_time = catalog_event_time + event_dt`.

## Install

```bash
cd /path/to/ADTomo_hz
pip install -r requirement.txt
python setup.py build_ext --inplace
pip install -e . --no-build-isolation
python -m pytest -q
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

Each test file is directly executable; it invokes its own pytest collection.
`tests/conftest.py` adds the adjacent source checkout to Python's import path.

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
grid = ForwardGrid(station, events, model, spacing=5.0)
phase_time = predict_phase_times(model, grid, "P", event_dt)
loss = ((phase_time - observed_phase_dt) ** 2).mean()
loss.backward()
```
