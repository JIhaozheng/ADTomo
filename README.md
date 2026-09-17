# ADTomo

A differentiable travel-time tomography framework based on the eikonal equation.

## Install

```bash
pip install -r requirement.txt
python setup.py build_ext --inplace
pip install -e . --no-build-isolation
```

## Synthetic example

```bash
cd examples

python 00_gen_velocity.py
python 01_gen_stations.py
python 02_gen_events.py
python 03_gen_picks.py
python 04_inversion.py
```

Results are written to `examples/results/`.

## Parallel example

```bash
cd examples
torchrun --standalone --nproc_per_node=2 parallel/inversion_ddp.py
```
