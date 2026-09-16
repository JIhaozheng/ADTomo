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


def solve_eikonal3d(velocity_xyz, source_xyz, spacing):
    """Solve the local 3-D eikonal equation for velocity in km/s.

    Local fields and source locations both use ``(x_local, y_local, z_local)``
    = ``(East, North, Down)``.
    """
    source_xyz = torch.as_tensor(source_xyz, dtype=velocity_xyz.dtype, device=velocity_xyz.device)
    slowness_xyz = (1.0 / velocity_xyz).contiguous()
    return _Eikonal3DFunction.apply(slowness_xyz, float(spacing), *source_xyz.tolist())
