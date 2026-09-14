"""Autograd wrapper for the retained two-dimensional CPU eikonal kernel."""

import torch

import eikonal2d_op


class _Eikonal2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, slowness, spacing, x, y):
        traveltime = eikonal2d_op.forward(slowness, spacing, x, y)
        ctx.save_for_backward(traveltime, slowness)
        ctx.spacing = spacing
        ctx.source = (x, y)
        return traveltime

    @staticmethod
    def backward(ctx, grad_output):
        traveltime, slowness = ctx.saved_tensors
        return eikonal2d_op.backward(grad_output.contiguous(), traveltime, slowness, ctx.spacing, *ctx.source), None, None, None


def solve_eikonal2d(velocity_yx, source_xy, spacing):
    """Solve a 2-D eikonal field; public array order is ``(y, x)``."""
    if velocity_yx.device.type != "cpu" or velocity_yx.dtype != torch.float64:
        raise ValueError("the retained eikonal kernels require CPU torch.float64 tensors")
    if velocity_yx.ndim != 2 or min(velocity_yx.shape) < 2:
        raise ValueError("velocity must have shape (y, x) with at least two nodes per axis")
    if not torch.isfinite(velocity_yx).all() or torch.any(velocity_yx <= 0):
        raise ValueError("velocity must be finite and positive")
    if not isinstance(spacing, (float, int)) or spacing <= 0:
        raise ValueError("spacing must be one positive isotropic scalar in km")
    source = torch.as_tensor(source_xy, dtype=velocity_yx.dtype, device=velocity_yx.device)
    upper = torch.tensor([velocity_yx.shape[1] - 1, velocity_yx.shape[0] - 1], dtype=velocity_yx.dtype)
    if source.shape != (2,) or not torch.all(source >= 0) or not torch.all(source < upper):
        raise ValueError("source must lie inside a forward-grid cell")
    tt_xy = _Eikonal2DFunction.apply((1.0 / velocity_yx).T.contiguous(), float(spacing), *[float(v) for v in source])
    return tt_xy.T
