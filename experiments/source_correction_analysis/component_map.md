# Production backward component map

The production source is `adtomo/eikonal/Eikonal3D.cpp`. Its forward source
treatment initializes the eight corners of the source cell with a Simpson rule:
source-interpolated slowness, cell-mean slowness, corner slowness, and physical
source-to-corner distance. FSM sweeps then leave those eight nodes pinned.

```text
Simpson source-corner initialization + pinned source corners
    |
    +-- SOURCE_STENCIL (S)
    |     adjderivonsource() supplies normal-projected derivatives at the
    |     eight corners; solve_adjoint_fsm_0f_src(..., use_source_fix=true)
    |     uses them only in the continuous-adjoint stencil.
    |
    +-- BULK (always present)
    |     lambda from the FSM adjoint produces lambda * slowness * h^3 at all
    |     nodes, including the source corners before any overwrite.
    |
    +-- SOURCE_GRADIENT (G)
          apply_source_simpson_grad_from_res() overwrites only the eight
          source-corner slowness gradients with the analytical derivative of
          the Simpson initialization.
              |
              +-- LOCAL_LU (L, only meaningful with G)
                    patch_lu_corner_res_3d() constructs a <=8 unknown dense
                    residual system over a one-cell source patch and replaces
                    the corner entries of lam_scaled before G consumes them.
```

`LOCAL_LU` has no observable mathematical output without `SOURCE_GRADIENT`:
the local solve produces `res`, and `res` is used only by the Simpson-gradient
overwrite. Therefore `L=1, G=0` combinations are reported as **N/A** rather
than fabricated as separate backward operators.

Affected nodes: S changes adjoint coefficients at eight source corners; L
solves for eight source-corner residual entries while coupling to a one-cell
ring; G overwrites eight source-corner entries of `dJ/ds`; bulk is global.
