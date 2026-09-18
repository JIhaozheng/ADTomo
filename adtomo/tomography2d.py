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


def predict_travel_times_2d(model, grid, phase, event_loc=None, event_indices=None):
    """Travel times for P or S events on one station's cached (y, x) section.

    Pass ``event_loc`` (``(N, 3)`` lon/lat/depth, possibly trainable) to sample
    the travel-time field at live event positions instead of the positions the
    grid was built with; ``event_indices`` then selects rows of ``event_loc``.
    """
    velocity = grid.sample_model({"P": model.vp, "S": model.vs}[phase.upper()])
    # eikonal2d_op works on an (x, y) layout; the grid stores fields as (y, x).
    slowness = (1.0 / velocity).T.contiguous()
    source = grid.station_index
    traveltime = _Eikonal2D.apply(slowness, grid.spacing, *source.tolist()).T
    if event_loc is None:
        return grid.sample_events(traveltime, event_indices=event_indices)
    if event_indices is not None:
        event_loc = event_loc[event_indices]
    return grid.sample_events(traveltime, index=grid.index_from_spherical(event_loc))


def smoothness_1d(field, depth):
    """Mean squared physical depth gradient of a depth-only field."""
    dz_km = depth[1:] - depth[:-1]
    return ((field[1:] - field[:-1]) / dz_km).square().mean()


class Tomography2D(nn.Module):
    """Arrival-time objective over a 1-D velocity model and trainable event parameters.

    ``event_loc`` holds absolute (lon, lat, depth) per event; the origin time is
    ``event_time_initial + event_time_correction`` so the trainable value stays
    small. Freeze the velocity model (``VelocityModel1D(..., trainable=False)``)
    for relocation only, or freeze the events (``trainable_location=False,
    trainable_time=False``) for velocity only; train everything for a joint
    inversion. Station groups are ``(grid, [(phase, event_indices,
    observed_phase_time), ...])`` with ``event_indices`` indexing ``event_loc``.
    """

    def __init__(
        self,
        model,
        event_loc,
        event_time,
        trainable_location=True,
        trainable_time=True,
        lambda_vp=0.0,
        lambda_vs=0.0,
        alpha_vp=0.0,
        alpha_vs=0.0,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("vp0", model.vp.detach().clone())
        self.register_buffer("vs0", model.vs.detach().clone())
        event_loc = torch.as_tensor(event_loc, dtype=torch.float64).detach().reshape(-1, 3)
        event_time = torch.as_tensor(event_time, dtype=torch.float64).detach().reshape(-1)
        self.event_loc = nn.Parameter(event_loc.clone(), requires_grad=trainable_location)
        self.register_buffer("event_time_initial", event_time.clone())
        self.event_time_correction = nn.Parameter(torch.zeros_like(event_time), requires_grad=trainable_time)
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

    @property
    def event_time(self):
        return self.event_time_initial + self.event_time_correction

    def forward(self, station_groups, data_scale=None):
        """Return the objective for station groups (see :class:`~adtomo.tomography3d.Tomography`)."""
        residuals = []
        for grid, phase_groups in station_groups:
            for phase, event_indices, observed_phase_time in phase_groups:
                travel_time = predict_travel_times_2d(
                    self.model, grid, phase, event_loc=self.event_loc, event_indices=event_indices
                )
                predicted = self.event_time[event_indices] + travel_time
                residuals.append(predicted - observed_phase_time)
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
        loss = data_loss + regularization_loss
        self.data_sum = data_sum.detach()
        self.data_loss = data_loss.detach()
        self.smooth_vp = smooth_vp.detach()
        self.smooth_vs = smooth_vs.detach()
        self.damp_vp = damp_vp.detach()
        self.damp_vs = damp_vs.detach()
        self.regularization_loss = regularization_loss.detach()
        self.total_loss = loss.detach()
        return loss
