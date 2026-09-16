# Component-level 3-D source-correction analysis

This diagnostic experiment builds one temporary extension with experiment-only
`SOURCE_STENCIL`, `LOCAL_LU`, and `SOURCE_GRADIENT` switches. Read
`component_map.md` before interpreting output. The retained LU path is the
independent reference for the production 3-D explicit source-corner balance.

Run a smoke suite with `MPLBACKEND=Agg python -m experiments.source_correction_analysis.run --quick`.
Run the complete isolated factorial suite by omitting `--quick`.

`LOCAL_LU` without `SOURCE_GRADIENT` is intentionally N/A: its local residual
has no output unless consumed by the Simpson gradient overwrite.
