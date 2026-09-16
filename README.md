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
- local physical coordinates: `(x_local, y_local, z_local)` = `(East, North, Down)`
- local scalar-field tensor order: `(z_local, y_local, x_local)` = `(Down, North, East)`
- retained C++ kernels: CPU-only, `torch.float64`, and one isotropic spacing

Each station defines the top of its local forward grid at `z = 0`. Horizontal
padding is symmetric; vertical padding is applied only below the event region.

The local Python and C++ field layouts are both DNE: a contiguous C++ buffer
uses `flat_index = x + nx * (y + ny * z)`, so East is the fastest-varying
axis. Source and event positions remain physical END coordinates. For PyTorch
`grid_sample`, query coordinates are consequently `[East, North, Down]` while
the sampled local field has DNE tensor order. None of these local conventions
apply to the global spherical model.

## Workflow assumptions

ADTomo is controlled research code. Provide regular increasing longitude,
latitude, and depth axes; positive Vp/Vs; unique station/event IDs; and picks
that reference those catalogs. Stations and events must lie inside the global
model domain. The retained eikonal kernels require CPU `torch.float64`,
positive isotropic spacing, positive velocity, and a source inside the local
forward grid.

## Catalog times

The catalog preserves ISO origin and phase timestamps. Every pick is converted
to its event-relative observed time:

```text
phase_dt = phase_time - catalog_event_time
```

With fixed catalog origin times, the modeled time is:

```text
predicted_phase_dt = travel_time
```

Event-origin corrections belong to a future joint event/velocity inversion.

## Velocity objective

`Tomography` combines the global arrival-time MSE with a physical
spatial-gradient penalty on velocity perturbations relative to the initial
model:

```text
J = (1/N) sum_i (predicted_phase_dt_i - observed_phase_dt_i)^2
    + lambda_vp ||grad_sph(Vp - Vp0)||^2
    + lambda_vs ||grad_sph(Vs - Vs0)||^2
    + alpha_vp ||Vp - Vp0||^2
    + alpha_vs ||Vs - Vs0||^2
```

`Vp0` and `Vs0` are fixed buffers captured when the objective is constructed.
With r = R_Earth - depth, the mean squared gradient uses
`(dm/dd)^2 + (dm/dphi / r)^2 + (dm/dlambda / (r cos(phi)))^2`.
Thus depth differences use km directly; latitude uses r dphi and longitude
uses r cos(phi) dlambda. All weights default to zero, preserving the
unregularized inversion. `lambda_vp` and `lambda_vs` are spherical-gradient
smoothness weights; `alpha_vp` and `alpha_vs` are damping weights toward the
initial model. Set `ADTOMO_LAMBDA_VP`, `ADTOMO_LAMBDA_VS`, `ADTOMO_ALPHA_VP`,
and `ADTOMO_ALPHA_VS` when running the synthetic inversion to compare a
regularized case.

## Install

```bash
cd /path/to/ADTomo
pip install -r requirement.txt
python setup.py build_ext --inplace
pip install -e . --no-build-isolation
```

The build compiles only `eikonal2d_op` and `eikonal3d_op`. A compiler with
C++20 support is required.

To run one test script from the test directory itself, build the extensions
once, then use:

```bash
cd /path/to/ADTomo/tests
python test_eikonal.py
python test_grid.py
python test_gradient.py
python benchmark_interpolation.py
```

Each test file is directly executable after the editable installation. They
are scientific validation scripts, not framework-collected test modules. They
save diagnostic figures to `tests/figures/eikonal.png`,
`tests/figures/grid_interpolation.png`, and
`tests/figures/taylor_remainders.png`. The three plotting scripts also call
`plt.show()`, so the figures appear when an interactive Matplotlib backend is
available. These generated PNG files are ignored.

`benchmark_interpolation.py` compares the retained `grid_sample` event
interpolation with vectorized manual trilinear interpolation on CPU float64.
It verifies values and travel-time-field gradients, then reports forward and
backward timings for 1 through 10,000 events. `grid_sample` is retained because
it is the clearer and faster path in this benchmark.

## Synthetic workflow

```bash
cd examples
python 00_gen_velocity.py
python 01_gen_stations.py
python 02_gen_events.py
python 03_gen_picks.py
python 04_inversion.py
```

Catalog files are written to `examples/data/`; the inversion writes one final
model and `inversion_progress.png` to `examples/results/`.

For station-group CPU DDP, run the same inversion on two ranks from the
`examples` directory:

```bash
torchrun --standalone --nproc_per_node=2 parallel/inversion_ddp.py
torchrun --standalone --nproc_per_node=2 parallel/inversion_ddp.py --validate
```

Each rank owns `groups[rank::world_size]`. Its local squared-residual sum is
scaled by `world_size / N`, so DDP's averaged gradient equals the serial MSE;
the identical full-model regularization is evaluated on every rank.

The complete forward path stays visible:

```python
grid = ForwardGrid(station_spherical, events_spherical, model, spacing=5.0)
predicted_phase_dt = predict_travel_times(model, grid, "P")
loss = ((predicted_phase_dt - observed_phase_dt) ** 2).mean()
loss.backward()
```
