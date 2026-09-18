"""2-D eikonal tomography for a depth-only velocity model with trainable event parameters.

A spherically symmetric model ``v(depth)`` keeps every station-event ray in
the great-circle plane through Earth's center, so each station's forward
problem is a 2-D Cartesian eikonal solve on that section
(:class:`~adtomo.grid.ForwardGrid2D`, station at the origin as the source,
``v_2d(x, y) = v[d(x, y)]`` with ``d(x, y) = R - sqrt(x^2 + (R - d_s - y)^2)``).
Event locations and origin-time corrections are trainable tensors whose
gradients flow through the grid's differentiable coordinate mapping and
travel-time interpolation.
"""

import math

import eikonal2d_op
import torch
import torch.nn as nn


class _Eikonal2D(torch.autograd.Function):
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
        grad_slowness = eikonal2d_op.backward(
            grad_output.contiguous(), traveltime, slowness, ctx.spacing, *ctx.source
        )
        return grad_slowness, None, None, None


def _check_solver_inputs(velocity, source, spacing):
    """Reject inputs the C++ kernel would otherwise accept silently (it clamps the source cell)."""
    if velocity.ndim != 2 or min(velocity.shape) < 2:
        raise ValueError("velocity must have (y, x) shape with at least two nodes per axis")
    if velocity.device.type != "cpu" or velocity.dtype != torch.float64:
        raise ValueError("2-D eikonal solver requires a CPU float64 velocity field")
    if not torch.isfinite(velocity).all() or torch.any(velocity <= 0):
        raise ValueError("velocity must be finite and positive")
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    if source.numel() != 2:
        raise ValueError("source must contain local (x, y) indices")
    upper = torch.tensor([velocity.shape[1] - 1, velocity.shape[0] - 1], dtype=velocity.dtype)
    if not torch.isfinite(source).all() or torch.any(source < 0) or torch.any(source >= upper):
        raise ValueError("source must lie inside a valid 2-D source cell")


def predict_travel_times_2d(model, grid, phase, events_spherical):
    """Travel times from one station's section to live event positions.

    ``events_spherical`` is ``(N, 3)`` lon/lat/depth and may be a trainable
    tensor: the station is the fixed eikonal source, and event gradients come
    from the differentiable geometry and bilinear interpolation, never from the
    C++ source position.
    """
    velocity = grid.sample_model({"P": model.vp, "S": model.vs}[phase.upper()], model.depth)
    _check_solver_inputs(velocity, grid.station_index, grid.spacing)
    # eikonal2d_op works on an (x, y) layout; the grid stores fields as (y, x).
    slowness = (1.0 / velocity).T.contiguous()
    traveltime = _Eikonal2D.apply(slowness, grid.spacing, *grid.station_index.tolist()).T
    return grid.sample_events(traveltime, events_spherical)


def smoothness_1d(field, depth):
    """Mean squared physical depth gradient of a depth-only field."""
    dz_km = depth[1:] - depth[:-1]
    return ((field[1:] - field[:-1]) / dz_km).square().mean()


class Tomography2D(nn.Module):
    """Arrival-time objective over a 1-D velocity model and trainable event parameters.

    ``event_loc`` holds absolute (lon, lat, depth) per event, initialized from
    the catalog, and ``event_time_correction`` the origin-time shift, so
    ``t_pred - t0_initial = dt0 + T`` against ``observed_phase_dt = phase_time
    - t0_initial``. Toggle ``requires_grad`` on ``model.vp``, ``model.vs``,
    ``event_loc``, ``event_time_correction`` to choose what is inverted.
    Station groups are ``(grid, [(phase, event_indices, observed_phase_dt),
    ...])`` with ``event_indices`` indexing ``event_loc``.
    """

    def __init__(self, model, event_loc, lambda_vp=0.0, lambda_vs=0.0, alpha_vp=0.0, alpha_vs=0.0):
        super().__init__()
        self.model = model
        self.register_buffer("vp0", model.vp.detach().clone())
        self.register_buffer("vs0", model.vs.detach().clone())
        event_loc = torch.as_tensor(event_loc, dtype=torch.float64).detach().reshape(-1, 3).contiguous()
        self.event_loc = nn.Parameter(event_loc.clone())
        self.event_time_correction = nn.Parameter(torch.zeros(len(event_loc), dtype=torch.float64))
        self.lambda_vp = lambda_vp
        self.lambda_vs = lambda_vs
        self.alpha_vp = alpha_vp
        self.alpha_vs = alpha_vs
        self.data_sum = None
        self.data_loss = None
        self.smooth_vp = None
        self.smooth_vs = None
        self.damp_vp = None
        self.damp_vs = None
        self.regularization_loss = None
        self.total_loss = None

    def forward(self, station_groups, data_scale=None, regularization_scale=1.0):
        """Return the objective (scaling conventions as in :class:`~adtomo.tomography3d.Tomography`)."""
        residuals = []
        for grid, phase_groups in station_groups:
            for phase, event_indices, observed_phase_dt in phase_groups:
                travel_time = predict_travel_times_2d(self.model, grid, phase, self.event_loc[event_indices])
                predicted = travel_time + self.event_time_correction[event_indices]
                residuals.append(predicted - observed_phase_dt)
        residual = torch.cat(residuals)
        data_sum = residual.square().sum()
        data_loss = data_sum / residual.numel() if data_scale is None else data_scale * data_sum
        dvp = self.model.vp - self.vp0
        dvs = self.model.vs - self.vs0
        smooth_vp = smoothness_1d(dvp, self.model.depth)
        smooth_vs = smoothness_1d(dvs, self.model.depth)
        damp_vp = dvp.square().mean()
        damp_vs = dvs.square().mean()
        regularization_loss = (
            self.lambda_vp * smooth_vp
            + self.lambda_vs * smooth_vs
            + self.alpha_vp * damp_vp
            + self.alpha_vs * damp_vs
        )
        loss = data_loss + regularization_scale * regularization_loss
        self.data_sum = data_sum.detach()
        self.data_loss = data_loss.detach()
        self.smooth_vp = smooth_vp.detach()
        self.smooth_vs = smooth_vs.detach()
        self.damp_vp = damp_vp.detach()
        self.damp_vs = damp_vs.detach()
        self.regularization_loss = regularization_loss.detach()
        self.total_loss = loss.detach()
        return loss
