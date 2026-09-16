# 3-D source-correction experiment

This isolated experiment compares three backward implementations for the
production 3-D eikonal forward solver. It does not edit or import a replacement
for `adtomo/eikonal/Eikonal3D.cpp`.

From the `ADTomo_hz` repository root, run:

```bash
MPLBACKEND=Agg python -m experiments.source_correction.run
```

The command builds three temporary Torch extensions in Torch's cache and writes
a timestamped directory below `experiments/source_correction/results/`. That
directory contains a manifest, all Taylor remainders and slopes, compact tables,
plots, runtime measurements, and the evidence-based recommendation.

The variants have one intentional difference each:

- `full` is the frozen pre-refactor production baseline. Its forward section
  remains byte-identical to the other experiment variants and is checked
  numerically against the refactored production forward.
- `no_lu` uses the source-aware continuous adjoint and the direct
  Simpson/source-corner gradient, but supplies `lam_scaled` directly instead of
  solving the local dense LU patch.
- `bulk_only` disables the source-aware stencil and uses only the continuous
  adjoint bulk slowness gradient.

Run the focused contract and smoke test with:

```bash
MPLBACKEND=Agg python tests/test_source_correction_experiment.py
```

Production `Eikonal3D.cpp`, `setup.py`, `ForwardGrid`, and production wrappers
are deliberately outside this experiment's write scope.
