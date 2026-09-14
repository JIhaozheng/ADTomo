"""Autograd wrapper for the retained three-dimensional CPU eikonal kernel."""

import torch

import eikonal3d_op


def _check(velocity, source, spacing):
    if velocity.device.type != "cpu" or velocity.dtype != torch.float64:
        raise ValueError("the retained eikonal kernels require CPU torch.float64 tensors")
    if velocity.ndim != 3 or min(velocity.shape) < 2:
        raise ValueError("local velocity must have shape (z_local, y_local, x_local) with at least two nodes per axis")
    if not torch.isfinite(velocity).all() or torch.any(velocity <= 0):
        raise ValueError("velocity must be finite and positive")
    if not isinstance(spacing, (float, int)) or spacing <= 0:
        raise ValueError("spacing must be one positive isotropic scalar in km")
    source = torch.as_tensor(source, dtype=velocity.dtype, device=velocity.device)
    if source.shape != (3,):
        raise ValueError("source must be fractional (x, y, z) indices")
    nx, ny, nz = velocity.shape[2], velocity.shape[1], velocity.shape[0]
    upper = torch.tensor([nx - 1, ny - 1, nz - 1], dtype=velocity.dtype)
    if not torch.all(source >= 0) or not torch.all(source < upper):
        raise ValueError("source must lie inside a forward-grid cell")
    return source


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
    source = _check(velocity_zyx, source_xyz, spacing)
    slowness_xyz = (1.0 / velocity_zyx).permute(2, 1, 0).contiguous()
    tt_xyz = _Eikonal3DFunction.apply(slowness_xyz, float(spacing), *[float(v) for v in source])
    return tt_xyz.permute(2, 1, 0)
