"""Autograd wrapper for the retained three-dimensional CPU eikonal kernel."""

import math

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
    if velocity_zyx.device.type != "cpu" or velocity_zyx.dtype != torch.float64:
        raise ValueError("velocity must be CPU float64")
    if velocity_zyx.ndim != 3 or min(velocity_zyx.shape) < 2:
        raise ValueError("velocity must be a (z_local, y_local, x_local) field with at least two nodes per axis")
    if not torch.isfinite(velocity_zyx).all() or torch.any(velocity_zyx <= 0):
        raise ValueError("velocity must be finite and positive")
    try:
        spacing = float(spacing)
    except (TypeError, ValueError) as error:
        raise ValueError("spacing must be finite and positive") from error
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    source_xyz = torch.as_tensor(source_xyz, dtype=velocity_zyx.dtype, device=velocity_zyx.device)
    nx, ny, nz = velocity_zyx.shape[2], velocity_zyx.shape[1], velocity_zyx.shape[0]
    upper = torch.tensor([nx - 1, ny - 1, nz - 1], dtype=velocity_zyx.dtype)
    if (
        source_xyz.shape != (3,)
        or not torch.isfinite(source_xyz).all()
        or not torch.all(source_xyz >= 0)
        or not torch.all(source_xyz < upper)
    ):
        raise ValueError("source must be finite (x_local, y_local, z_local) indices inside the forward grid")
    slowness_xyz = (1.0 / velocity_zyx).permute(2, 1, 0).contiguous()
    tt_xyz = _Eikonal3DFunction.apply(slowness_xyz, spacing, *source_xyz.tolist())
    return tt_xyz.permute(2, 1, 0)
