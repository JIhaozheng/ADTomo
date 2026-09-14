"""Autograd wrapper for the retained two-dimensional CPU eikonal kernel."""

import math

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
    """Solve a local 2-D eikonal field in ``(y_local, x_local)`` order."""
    if velocity_yx.device.type != "cpu" or velocity_yx.dtype != torch.float64:
        raise ValueError("velocity must be CPU float64")
    if velocity_yx.ndim != 2 or min(velocity_yx.shape) < 2:
        raise ValueError("velocity must be a (y_local, x_local) field with at least two nodes per axis")
    if not torch.isfinite(velocity_yx).all() or torch.any(velocity_yx <= 0):
        raise ValueError("velocity must be finite and positive")
    spacing = float(spacing)
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    source = torch.as_tensor(source_xy, dtype=velocity_yx.dtype, device=velocity_yx.device)
    upper = torch.tensor([velocity_yx.shape[1] - 1, velocity_yx.shape[0] - 1], dtype=velocity_yx.dtype)
    if source.shape != (2,) or not torch.isfinite(source).all() or not torch.all(source >= 0) or not torch.all(source < upper):
        raise ValueError("source must be finite (x_local, y_local) indices inside the forward grid")
    slowness_xy = (1.0 / velocity_yx).T.contiguous()
    tt_xy = _Eikonal2DFunction.apply(slowness_xy, spacing, *source.tolist())
    return tt_xy.T
