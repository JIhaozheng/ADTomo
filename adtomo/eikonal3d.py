"""Autograd wrapper for the retained three-dimensional CPU eikonal kernel."""

import torch

import eikonal3d_op


class _Eikonal3DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, slowness, spacing, x, y, z):
        traveltime = eikonal3d_op.forward(slowness, spacing, x, y, z)
        ctx.save_for_backward(traveltime, slowness)
        ctx.spacing = spacing
        ctx.source = (x, y, z)
        return traveltime

    @staticmethod
    def backward(ctx, grad_output):
        traveltime, slowness = ctx.saved_tensors
        grad_slowness = eikonal3d_op.backward(
            grad_output.contiguous(), traveltime, slowness, ctx.spacing, *ctx.source
        )
        return grad_slowness, None, None, None, None


def solve_eikonal3d(velocity, source, spacing):
    """Solve the 3-D eikonal equation on a local ``(z, y, x)`` velocity field."""
    source = torch.as_tensor(source, dtype=velocity.dtype, device=velocity.device)
    slowness = (1.0 / velocity).contiguous()
    return _Eikonal3DFunction.apply(slowness, float(spacing), *source.tolist())
