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


def solve_eikonal3d(velocity_zyx, source_xyz, spacing):
    """Solve the local 3-D eikonal equation for velocity in km/s.

    Python local fields use ``(z_local, y_local, x_local)`` = ``(Down, North,
    East)``. The retained kernel uses physical ``(x_local, y_local, z_local)``
    = ``(East, North, Down)``; that implementation detail is isolated here.
    """
    source_xyz = torch.as_tensor(source_xyz, dtype=velocity_zyx.dtype, device=velocity_zyx.device)
    slowness_xyz = (1.0 / velocity_zyx).permute(2, 1, 0).contiguous()
    tt_xyz = _Eikonal3DFunction.apply(slowness_xyz, float(spacing), *source_xyz.tolist())
    return tt_xyz.permute(2, 1, 0)
