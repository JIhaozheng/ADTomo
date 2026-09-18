"""Small functions connecting a 3-D velocity model, station grid, and observations."""

import eikonal3d_op
import torch
import torch.nn as nn


class _Eikonal3D(torch.autograd.Function):
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


def predict_travel_times(model, grid, phase, event_indices=None, index=None):
    """Travel times for P or S events on one station's cached forward grid.

    Pass ``index`` (from ``grid.index_from_spherical``) to sample at live,
    possibly trainable event positions instead of the cached build-time ones.
    """
    velocity = grid.sample_model({"P": model.vp, "S": model.vs}[phase.upper()])
    slowness = 1.0 / velocity
    source = grid.station_index
    traveltime = _Eikonal3D.apply(slowness, grid.spacing, *source.tolist())
    return grid.sample_events(traveltime, event_indices=event_indices, index=index)


def smoothness(field, lon, lat, depth):
    """Mean squared physical gradients of a (depth, latitude, longitude) field."""
    R_EARTH = 6371.0 # km
    radius = R_EARTH - depth
    dz_km = depth[1:] - depth[:-1]
    dlat = torch.deg2rad(lat[1:] - lat[:-1])
    dlon = torch.deg2rad(lon[1:] - lon[:-1])
    grad_depth = (field[1:, :, :] - field[:-1, :, :]) / dz_km[:, None, None]
    grad_lat = (field[:, 1:, :] - field[:, :-1, :]) / (radius[:, None, None] * dlat[None, :, None])
    grad_lon = (field[:, :, 1:] - field[:, :, :-1]) / (
        radius[:, None, None] * torch.cos(torch.deg2rad(lat))[None, :, None] * dlon[None, None, :]
    )
    return grad_depth.square().mean() + grad_lat.square().mean() + grad_lon.square().mean()


class Tomography(nn.Module):
    """Arrival-time objective for a 3-D model, optionally with trainable event parameters.

    Without ``event_loc`` each phase group's ``event_indices`` index the
    station grid's cached build-time events. With ``event_loc`` (``(N, 3)``
    initial lon/lat/depth) the module also holds ``event_time_correction``
    (``N`` seconds), ``event_indices`` index that catalog, and every event is
    re-mapped live through the grid's differentiable coordinate path so
    hypocenters can be relocated: ``t_pred - t0_initial = dt0 + T`` against
    ``observed_phase_dt = phase_time - t0_initial``. Toggle ``requires_grad``
    on ``model.vp``, ``model.vs``, ``event_loc``, ``event_time_correction`` to
    choose what is inverted.
    """

    def __init__(self, model, lambda_vp=0.0, lambda_vs=0.0, alpha_vp=0.0, alpha_vs=0.0, event_loc=None):
        super().__init__()
        self.model = model
        self.register_buffer("vp0", model.vp.detach().clone())
        self.register_buffer("vs0", model.vs.detach().clone())
        if event_loc is None:
            self.event_loc = None
            self.event_time_correction = None
        else:
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
        """Return the tomography objective for station groups.

        In DistributedDataParallel, pass ``world_size / total_observations``
        as ``data_scale`` so averaged rank gradients match serial MSE. With
        explicitly summed gradients instead, pass ``1 / total_observations``
        and ``regularization_scale=1 / world_size`` so the regularization is
        counted once.
        """
        residuals = []
        for grid, phase_groups in station_groups:
            for phase, event_indices, observed_phase_dt in phase_groups:
                if self.event_loc is None:
                    predicted = predict_travel_times(self.model, grid, phase, event_indices)
                else:
                    index = grid.index_from_spherical(self.event_loc[event_indices])
                    predicted = predict_travel_times(self.model, grid, phase, index=index)
                    predicted = predicted + self.event_time_correction[event_indices]
                residuals.append(predicted - observed_phase_dt)
        residual = torch.cat(residuals)
        data_sum = residual.square().sum()
        data_loss = data_sum / residual.numel() if data_scale is None else data_scale * data_sum
        dvp = self.model.vp - self.vp0
        dvs = self.model.vs - self.vs0
        smooth_vp = smoothness(dvp, self.model.lon, self.model.lat, self.model.depth)
        smooth_vs = smoothness(dvs, self.model.lon, self.model.lat, self.model.depth)
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
