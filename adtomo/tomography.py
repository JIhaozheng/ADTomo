"""Small functions connecting a velocity model, station grid, and observations."""

import torch
import torch.nn as nn

from .eikonal3d import solve_eikonal3d


def predict_travel_times(model, grid, phase, event_indices=None):
    """Travel times for P or S events on one station's cached forward grid."""
    velocity = grid.sample({"P": model.vp, "S": model.vs}[phase.upper()])
    traveltime = solve_eikonal3d(velocity, grid.station_index, grid.spacing)
    return grid.sample_events(traveltime, event_indices=event_indices)


def smoothness(model):
    """Mean squared first differences of a (depth, latitude, longitude) field."""
    dz = model[1:, :, :] - model[:-1, :, :]
    dy = model[:, 1:, :] - model[:, :-1, :]
    dx = model[:, :, 1:] - model[:, :, :-1]
    return dz.square().mean() + dy.square().mean() + dx.square().mean()


class Tomography(nn.Module):
    """Arrival-time objective with optional smoothness of velocity perturbations."""

    def __init__(self, model, lambda_vp=0.0, lambda_vs=0.0):
        super().__init__()
        self.model = model
        self.register_buffer("vp0", model.vp.detach().clone())
        self.register_buffer("vs0", model.vs.detach().clone())
        self.lambda_vp = lambda_vp
        self.lambda_vs = lambda_vs
        self.data_loss = None
        self.reg_vp = None
        self.reg_vs = None

    def forward(self, station_groups, data_scale=1.0):
        residuals = []
        for grid, phase_groups in station_groups:
            for phase, grid_event_indices, observed_phase_dt in phase_groups:
                predicted = predict_travel_times(self.model, grid, phase, grid_event_indices)
                residuals.append(predicted - observed_phase_dt)
        residual = torch.cat(residuals)
        data_loss = residual.square().mean()
        reg_vp = smoothness(self.model.vp - self.vp0)
        reg_vs = smoothness(self.model.vs - self.vs0)
        loss = data_scale * data_loss + self.lambda_vp * reg_vp + self.lambda_vs * reg_vs
        self.data_loss = data_loss.detach()
        self.reg_vp = reg_vp.detach()
        self.reg_vs = reg_vs.detach()
        return loss
