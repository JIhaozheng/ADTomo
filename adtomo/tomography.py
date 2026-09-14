"""Small functions connecting a velocity model, station grid, and observations."""

import torch
import torch.nn as nn

from .eikonal3d import solve_eikonal3d
from .coordinate import R_EARTH_KM


def predict_travel_times(model, grid, phase, event_indices=None):
    """Travel times for P or S events on one station's cached forward grid."""
    velocity = grid.sample({"P": model.vp, "S": model.vs}[phase.upper()])
    traveltime = solve_eikonal3d(velocity, grid.station_index, grid.spacing)
    return grid.sample_events(traveltime, event_indices=event_indices)


def smoothness(field, lon, lat, depth):
    """Mean squared physical gradients of a (depth, latitude, longitude) field."""
    radius = R_EARTH_KM - depth
    dz_km = depth[1:] - depth[:-1]
    dlat = torch.deg2rad(lat[1:] - lat[:-1])
    dlon = torch.deg2rad(lon[1:] - lon[:-1])
    grad_z = (field[1:, :, :] - field[:-1, :, :]) / dz_km[:, None, None]
    grad_y = (field[:, 1:, :] - field[:, :-1, :]) / (radius[:, None, None] * dlat[None, :, None])
    grad_x = (field[:, :, 1:] - field[:, :, :-1]) / (
        radius[:, None, None] * torch.cos(torch.deg2rad(lat))[None, :, None] * dlon[None, None, :]
    )
    return grad_z.square().mean() + grad_y.square().mean() + grad_x.square().mean()


class Tomography(nn.Module):
    """Arrival-time objective with smoothness and damping of velocity perturbations."""

    def __init__(self, model, lambda_vp=0.0, lambda_vs=0.0, alpha_vp=0.0, alpha_vs=0.0):
        super().__init__()
        self.model = model
        self.register_buffer("vp0", model.vp.detach().clone())
        self.register_buffer("vs0", model.vs.detach().clone())
        self.lambda_vp = lambda_vp
        self.lambda_vs = lambda_vs
        self.alpha_vp = alpha_vp
        self.alpha_vs = alpha_vs
        self.data_loss = None
        self.smooth_vp = None
        self.smooth_vs = None
        self.damp_vp = None
        self.damp_vs = None
        self.total_loss = None

    def forward(self, station_groups):
        residuals = []
        for grid, phase_groups in station_groups:
            for phase, grid_event_indices, observed_phase_dt in phase_groups:
                predicted = predict_travel_times(self.model, grid, phase, grid_event_indices)
                residuals.append(predicted - observed_phase_dt)
        residual = torch.cat(residuals)
        data_loss = residual.square().mean()
        dvp = self.model.vp - self.vp0
        dvs = self.model.vs - self.vs0
        smooth_vp = smoothness(dvp, self.model.lon, self.model.lat, self.model.depth)
        smooth_vs = smoothness(dvs, self.model.lon, self.model.lat, self.model.depth)
        damp_vp = dvp.square().mean()
        damp_vs = dvs.square().mean()
        loss = (
            data_loss
            + self.lambda_vp * smooth_vp
            + self.lambda_vs * smooth_vs
            + self.alpha_vp * damp_vp
            + self.alpha_vs * damp_vs
        )
        self.data_loss = data_loss.detach()
        self.smooth_vp = smooth_vp.detach()
        self.smooth_vs = smooth_vs.detach()
        self.damp_vp = damp_vp.detach()
        self.damp_vs = damp_vs.detach()
        self.total_loss = loss.detach()
        return loss
