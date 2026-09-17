# ADTomo

A differentiable travel-time tomography framework based on the eikonal equation.

## Install

```bash
pip install -r requirement.txt
python setup.py build_ext --inplace
pip install -e . --no-build-isolation
```

## Example

Generate the synthetic checkerboard dataset:

```bash
python examples/00_gen_velocity.py
python examples/01_gen_stations.py --num-stations 12
python examples/02_gen_events.py --num-events 50
python examples/03_gen_picks.py
```

Run the inversion:

```bash
bash examples/run_inversion.sh
```

Run it in parallel:

```bash
NPROC=4 bash examples/run_inversion.sh
```

For an HPC job, set inversion parameters through the environment:

```bash
ITERATIONS=100 LEARNING_RATE=0.01 NPROC=8 \
    bash examples/run_inversion.sh
```
