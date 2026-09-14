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
    station_lonlatdepth = torch.tensor([-120.0, 35.0, 0.0], dtype=torch.float64)
    event_lonlatdepth = torch.tensor([[-119.9, 35.1, 10.0]], dtype=torch.float64)
    grid = ForwardGrid(station_lonlatdepth, event_lonlatdepth, model, spacing=5.0)
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


def test_full_spherical_vp_taylor_remainder():
    lon = torch.arange(-120.8, -119.19, 0.1, dtype=torch.float64)
    lat = torch.arange(34.2, 35.81, 0.1, dtype=torch.float64)
    depth = torch.arange(-15.0, 50.1, 5.0, dtype=torch.float64)
    vp = torch.full((len(depth), len(lat), len(lon)), 6.0, dtype=torch.float64)
    vs = vp / 1.73
    model = VelocityModel(lon, lat, depth, vp, vs, trainable=True)
    station_lonlatdepth = torch.tensor([-120.0, 35.0, 0.0], dtype=torch.float64)
    catalog_event_lonlatdepth = torch.tensor(
        [[-119.9, 35.1, 10.0], [-120.1, 34.9, 12.0]], dtype=torch.float64
    )
    grid = ForwardGrid(station_lonlatdepth, catalog_event_lonlatdepth[1:], model, spacing=5.0)
    catalog_event_indices = torch.tensor([1], dtype=torch.long)
    grid_event_indices = torch.tensor([0], dtype=torch.long)
    event_dt = torch.tensor([-0.10, 0.05], dtype=torch.float64)
    direction = torch.linspace(-0.01, 0.01, model.vp.numel(), dtype=torch.float64).reshape_as(model.vp)

    def loss_for(candidate):
        predicted_phase_dt = predict_phase_times(
            candidate,
            grid,
            "P",
            event_dt[catalog_event_indices],
            event_indices=grid_event_indices,
        )
        return (predicted_phase_dt - 3.0).square().sum()

    loss = loss_for(model)
    loss.backward()
    derivative = (model.vp.grad * direction).sum().item()
    base_vp = model.vp.detach()
    epsilons = (1e-2, 5e-3, 2.5e-3, 1.25e-3)
    remainders = []
    for epsilon in epsilons:
        perturbed = VelocityModel(lon, lat, depth, base_vp + epsilon * direction, vs, trainable=False)
        remainder = abs(loss_for(perturbed).item() - loss.item() - epsilon * derivative)
        remainders.append(remainder)
    assert all(math.isfinite(remainder) and remainder > 0 for remainder in remainders)
    assert all(right < left for left, right in zip(remainders, remainders[1:]))
    slopes = [
        math.log(right / left) / math.log(next_epsilon / epsilon)
        for left, right, epsilon, next_epsilon in zip(remainders, remainders[1:], epsilons, epsilons[1:])
    ]
    assert 1.7 < sorted(slopes)[len(slopes) // 2] < 2.3


def main():
    test_eikonal_velocity_gradient_is_finite()
    test_eikonal_taylor_remainder_is_second_order()
    test_full_spherical_gradient_path()
    test_full_spherical_vp_taylor_remainder()
    print(f"{Path(__file__).name}: passed")


if __name__ == "__main__":
    main()
