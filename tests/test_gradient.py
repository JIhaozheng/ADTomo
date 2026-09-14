import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from adtomo import ForwardGrid, VelocityModel, predict_phase_times, solve_eikonal3d


def test_eikonal_velocity_gradient_is_finite():
    velocity = torch.full((4, 5, 6), 5.0, dtype=torch.float64, requires_grad=True)
    travel_time = solve_eikonal3d(velocity, (1.2, 1.3, 1.1), 1.0)
    travel_time[3, 4, 5].backward()
    assert velocity.grad is not None
    assert torch.isfinite(velocity.grad).all()
    assert velocity.grad.abs().sum() > 0


def test_eikonal_taylor_remainder_is_second_order():
    velocity = torch.full((4, 5, 6), 5.0, dtype=torch.float64, requires_grad=True)
    direction = torch.linspace(-0.2, 0.2, velocity.numel(), dtype=torch.float64).reshape_as(velocity)
    objective = solve_eikonal3d(velocity, (1.2, 1.3, 1.1), 1.0)[3, 4, 5]
    objective.backward()
    derivative = (velocity.grad * direction).sum().item()
    remainder = []
    for step in (1e-2, 1e-3):
        perturbed = solve_eikonal3d(velocity.detach() + step * direction, (1.2, 1.3, 1.1), 1.0)[3, 4, 5]
        remainder.append(abs(perturbed.item() - objective.item() - step * derivative))
    slope = math.log(remainder[1] / remainder[0]) / math.log(1e-3 / 1e-2)
    assert 1.8 < slope < 2.2


def test_full_spherical_gradient_path():
    lon = torch.arange(-120.8, -119.19, 0.1, dtype=torch.float64)
    lat = torch.arange(34.2, 35.81, 0.1, dtype=torch.float64)
    depth = torch.arange(-15.0, 50.1, 5.0, dtype=torch.float64)
    vp = torch.full((len(depth), len(lat), len(lon)), 6.0, dtype=torch.float64)
    model = VelocityModel(lon, lat, depth, vp, vp / 1.73, trainable=True)
    station = torch.tensor([-120.0, 35.0, 0.0], dtype=torch.float64)
    events = torch.tensor([[-119.9, 35.1, 10.0]], dtype=torch.float64)
    grid = ForwardGrid(station, events, model, spacing=5.0)
    prediction_p = predict_phase_times(model, grid, "P", torch.zeros(1, dtype=torch.float64))
    prediction_s = predict_phase_times(model, grid, "S", torch.zeros(1, dtype=torch.float64))
    loss = (prediction_p - 3.0).square().mean() + (prediction_s - 5.0).square().mean()
    loss.backward()
    assert model.vp.grad is not None
    assert torch.isfinite(model.vp.grad).all()
    assert model.vp.grad.abs().sum() > 0
    assert model.vs.grad is not None
    assert torch.isfinite(model.vs.grad).all()
    assert model.vs.grad.abs().sum() > 0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
