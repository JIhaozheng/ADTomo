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
    """Solve a local 2-D eikonal field in ``(y_local, x_local)`` order."""
    source = torch.as_tensor(source_xy, dtype=velocity_yx.dtype, device=velocity_yx.device)
    slowness_xy = (1.0 / velocity_yx).T.contiguous()
    tt_xy = _Eikonal2DFunction.apply(slowness_xy, float(spacing), *source.tolist())
    return tt_xy.T
