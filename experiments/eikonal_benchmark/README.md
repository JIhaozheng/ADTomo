# 3-D Eikonal benchmark

This isolated experiment freezes and compares the 3-D adjoint variants before
any production cleanup.  The frozen sources are:

| Variant | Provenance | Commit | SHA256 |
|---|---|---|---|
| `old_full` | ADTomo_hz last `S=1,L=1,G=1` solver | `70ba811` | `28277d7eb2dc99dbb3e7e84938a9021d5e00cb5e393f31ea3dc276b264b1a3f1` |
| `current` | production before cleanup | `8c69f04` | `6180c0b8aeb2a15c3e9507b01c158c5e7dd282c334a80976e835b261e12f0dec` |
| `weiqiang` | AI4EPS/ADTomo `Eikonal3D_xyz.cpp` | `18e317e990ab314a5421a3ad1a0a454f01523598` | `ddebcf14686c64cccc9a4e48fddbc885f528ab3c87e20ceb10d23e7fdc24d28f` |

The sources in `cpp/` are immutable benchmark inputs.  The forthcoming
`cleaned` candidate is created only after baseline correctness and timing
artifacts are recorded.  Generated artifacts are written under `results/` and
are ignored by Git.
