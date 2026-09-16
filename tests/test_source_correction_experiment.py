"""Focused contract and smoke test for the isolated source-correction experiment."""

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.source_correction.loader import PRODUCTION, VARIANTS, load_extensions, validate_sources
from experiments.source_correction_analysis.loader import load_extension as load_component_extension


def main():
    validate_sources()
    ops = load_extensions()
    f = torch.full((9, 10, 11), 1.0 / 6.0, dtype=torch.float64)
    source = (3.4, 4.5, 5.6)
    fields = {name: op.forward(f.contiguous(), 1.0, *source) for name, op in ops.items()}
    assert all(torch.equal(fields["full"], field) for field in fields.values())

    grad_u = torch.zeros_like(f)
    grad_u[7, 8, 9] = 1.0
    full = ops["full"].backward(grad_u, fields["full"], f, 1.0, *source)
    import eikonal3d_op

    production_field = eikonal3d_op.forward(f.contiguous(), 1.0, *source)
    production_grad = eikonal3d_op.backward(grad_u, production_field, f, 1.0, *source)
    assert torch.equal(fields["full"], production_field)
    # Production now uses the normal bulk stencil plus the exact source-cell
    # residual balance.  It must agree with the retained LU reference variant.
    component = load_component_extension()
    reference = component.backward_components(
        grad_u, fields["full"], f, 1.0, *source, False, True, True
    )
    assert torch.equal(production_grad, reference)
    # The source-aware-stencil legacy full path is retained only as an
    # experiment baseline; it agrees on this contract case as well.
    assert torch.equal(full, production_grad)

    # Aligned and fractional sources retain the frozen forward bit-for-bit;
    # the production explicit corner balance is numerically equivalent to LU.
    axis = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    xx, yy, zz = torch.meshgrid(axis, axis, axis, indexing="ij")
    smooth = 1.0 / (6.0 * (1.0 + 0.05 * torch.sin(6.283185307 * xx) * torch.cos(3.141592654 * yy) * torch.sin(6.283185307 * zz)))
    smooth_grad_u = torch.zeros_like(smooth)
    smooth_grad_u[7, 7, 7] = 1.0
    for checked_source in ((3.0, 3.0, 3.0), (3.4, 3.5, 3.6), (3.1, 3.15, 3.2)):
        frozen_u = ops["full"].forward(smooth.contiguous(), 1.0, *checked_source)
        current_u = eikonal3d_op.forward(smooth.contiguous(), 1.0, *checked_source)
        assert torch.equal(frozen_u, current_u)
        current_g = eikonal3d_op.backward(smooth_grad_u, current_u, smooth, 1.0, *checked_source)
        lu_g = component.backward_components(
            smooth_grad_u, frozen_u, smooth, 1.0, *checked_source, False, True, True
        )
        assert torch.allclose(current_g, lu_g, rtol=1e-14, atol=1e-15)
    assert torch.isfinite(ops["no_lu"].backward(grad_u, fields["no_lu"], f, 1.0, *source)).all()
    assert torch.isfinite(ops["bulk_only"].backward(grad_u, fields["bulk_only"], f, 1.0, *source)).all()
    print("test_source_correction_experiment.py: passed")


if __name__ == "__main__":
    main()
