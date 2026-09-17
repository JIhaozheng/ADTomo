# ADTomo

A differentiable travel-time tomography framework based on the eikonal equation.

## Install

```bash
pip install -r requirement.txt
python setup.py build_ext --inplace
pip install -e . --no-build-isolation
```

## Example

Serial:

```bash
bash examples/run.sh
```

Parallel:

```bash
NPROC=4 bash examples/run.sh
```

```bash
bash examples/run.sh --help
```
