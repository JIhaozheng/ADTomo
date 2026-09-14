# ADTomo

ADTomo is a small CPU implementation of differentiable spherical eikonal tomography using PyTorch and C++ fast-sweeping solvers.

## Install

```bash
cd /path/to/ADTomo
pip install -r requirement.txt
python setup.py build_ext --inplace
pip install -e . --no-build-isolation
```

The first build downloads Eigen and compiles the CPU eikonal solvers `eikonal2d_op` and `eikonal3d_op`.

The retained C++ kernels are CPU-only and use `torch.float64`.

## Synthetic workflow

Run the complete synthetic example from model generation to velocity inversion:

```bash
cd examples
python 00_gen_velocity.py
python 01_gen_stations.py
python 02_gen_events.py
python 03_gen_picks.py
python 04_forward.py
python 05_inversion.py
```

Synthetic data are written to `examples/data/`. The inversion result and `inversion_progress.png` are written to `examples/results/`.
