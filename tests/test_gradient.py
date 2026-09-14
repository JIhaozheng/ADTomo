import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from adtomo import ForwardGrid, Tomography, VelocityModel, predict_travel_times, smoothness, solve_eikonal3d


FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

# Isolated C++ eikonal Taylor check.
taylor_velocity = torch.full((4, 5, 6), 5.0, dtype=torch.float64, requires_grad=True)
taylor_direction = torch.linspace(-0.2, 0.2, taylor_velocity.numel(), dtype=torch.float64).reshape_as(taylor_velocity)
taylor_objective = solve_eikonal3d(taylor_velocity, (1.2, 1.3, 1.1), 1.0)[3, 4, 5]
taylor_objective.backward()
taylor_derivative = (taylor_velocity.grad * taylor_direction).sum().item()
kernel_remainders = []
for epsilon in (1e-2, 1e-3):
    perturbed = solve_eikonal3d(taylor_velocity.detach() + epsilon * taylor_direction, (1.2, 1.3, 1.1), 1.0)[3, 4, 5]
    kernel_remainders.append(abs(perturbed.item() - taylor_objective.item() - epsilon * taylor_derivative))
kernel_slope = math.log(kernel_remainders[1] / kernel_remainders[0]) / math.log(1e-3 / 1e-2)
assert 1.8 < kernel_slope < 2.2

lon = torch.arange(-120.8, -119.19, 0.1, dtype=torch.float64)
lat = torch.arange(34.2, 35.81, 0.1, dtype=torch.float64)
depth = torch.arange(-15.0, 50.1, 5.0, dtype=torch.float64)
vp = torch.full((len(depth), len(lat), len(lon)), 6.0, dtype=torch.float64)
station_lonlatdepth = torch.tensor([-120.0, 35.0, 0.0], dtype=torch.float64)
event_lonlatdepth = torch.tensor([[-119.9, 35.1, 10.0]], dtype=torch.float64)

# Full global-Vp chain: sampling, solver, event interpolation, loss.
taylor_model = VelocityModel(lon, lat, depth, vp, vp / 1.73, trainable=True)
taylor_grid = ForwardGrid(station_lonlatdepth, event_lonlatdepth, taylor_model, spacing=5.0)
global_direction = torch.linspace(-0.01, 0.01, taylor_model.vp.numel(), dtype=torch.float64).reshape_as(taylor_model.vp)


def full_phase_time_loss(candidate):
    predicted_phase_dt = predict_travel_times(candidate, taylor_grid, "P")
    return (predicted_phase_dt - 3.0).square().sum()


full_loss = full_phase_time_loss(taylor_model)
full_loss.backward()
assert taylor_model.vp.grad is not None
assert torch.isfinite(taylor_model.vp.grad).all()
assert taylor_model.vp.grad.abs().sum() > 0
full_derivative = (taylor_model.vp.grad * global_direction).sum().item()
base_vp = taylor_model.vp.detach()
epsilons = (1e-2, 5e-3, 2.5e-3, 1.25e-3)
full_remainders = []
for epsilon in epsilons:
    perturbed_model = VelocityModel(lon, lat, depth, base_vp + epsilon * global_direction, vp / 1.73, trainable=False)
    full_remainders.append(abs(full_phase_time_loss(perturbed_model).item() - full_loss.item() - epsilon * full_derivative))
assert all(math.isfinite(remainder) and remainder > 0 for remainder in full_remainders)
assert all(right < left for left, right in zip(full_remainders, full_remainders[1:]))
full_slopes = [
    math.log(right / left) / math.log(next_epsilon / epsilon)
    for left, right, epsilon, next_epsilon in zip(full_remainders, full_remainders[1:], epsilons, epsilons[1:])
]
assert 1.7 < sorted(full_slopes)[len(full_slopes) // 2] < 2.3

# Regularized tomography objective: smoothness applies to perturbations only.
objective_model = VelocityModel(lon, lat, depth, vp, vp / 1.73, trainable=True)
objective_grid = ForwardGrid(station_lonlatdepth, event_lonlatdepth, objective_model, spacing=5.0)
station_groups = [
    (
        objective_grid,
        [
            ("P", torch.tensor([0]), torch.tensor([3.0], dtype=torch.float64)),
            ("S", torch.tensor([0]), torch.tensor([5.0], dtype=torch.float64)),
        ],
    )
]
data_only = Tomography(objective_model)
data_only_loss = data_only(station_groups)
direct_residual = torch.cat(
    [
        predict_travel_times(objective_model, objective_grid, "P") - 3.0,
        predict_travel_times(objective_model, objective_grid, "S") - 5.0,
    ]
)
assert torch.allclose(data_only_loss, direct_residual.square().mean())
assert data_only.damp_vp.item() == 0.0
assert data_only.damp_vs.item() == 0.0
assert smoothness(
    torch.full_like(objective_model.vp, 0.2), objective_model.lon, objective_model.lat, objective_model.depth
).item() == 0.0

# A linear physical depth gradient has the same penalty on coarse and fine depth axes.
depth_gradient = 0.02
depth_coarse = torch.tensor([0.0, 5.0, 10.0], dtype=torch.float64)
depth_fine = torch.tensor([0.0, 2.5, 5.0, 7.5, 10.0], dtype=torch.float64)
lon_small = torch.tensor([-120.0, -119.9], dtype=torch.float64)
lat_small = torch.tensor([35.0, 35.1], dtype=torch.float64)
field_coarse = (depth_gradient * depth_coarse)[:, None, None].expand(-1, len(lat_small), len(lon_small))
field_fine = (depth_gradient * depth_fine)[:, None, None].expand(-1, len(lat_small), len(lon_small))
coarse_smoothness = smoothness(field_coarse, lon_small, lat_small, depth_coarse)
fine_smoothness = smoothness(field_fine, lon_small, lat_small, depth_fine)
assert torch.allclose(coarse_smoothness, torch.tensor(depth_gradient**2, dtype=torch.float64))
assert torch.allclose(fine_smoothness, coarse_smoothness)

constant_model = VelocityModel(lon, lat, depth, vp, vp / 1.73, trainable=True)
constant_tomography = Tomography(constant_model, alpha_vp=0.5, alpha_vs=0.25)
with torch.no_grad():
    constant_model.vp += 0.2
    constant_model.vs -= 0.1
constant_tomography(station_groups)
assert constant_tomography.smooth_vp.item() == 0.0
assert constant_tomography.smooth_vs.item() == 0.0
assert constant_tomography.damp_vp.item() > 0.0
assert constant_tomography.damp_vs.item() > 0.0

regularized = Tomography(
    objective_model, lambda_vp=0.5, lambda_vs=0.25, alpha_vp=0.125, alpha_vs=0.0625
)
with torch.no_grad():
    objective_model.vp[2, 3, 4] += 0.2
    objective_model.vs[2, 3, 4] -= 0.1
regularized_loss = regularized(station_groups)
assert regularized.smooth_vp.item() > 0.0
assert regularized.smooth_vs.item() > 0.0
assert regularized.damp_vp.item() > 0.0
assert regularized.damp_vs.item() > 0.0
assert torch.allclose(
    regularized_loss,
    regularized.data_loss
    + 0.5 * regularized.smooth_vp
    + 0.25 * regularized.smooth_vs
    + 0.125 * regularized.damp_vp
    + 0.0625 * regularized.damp_vs,
)
regularized_loss.backward()
assert torch.isfinite(objective_model.vp.grad).all() and objective_model.vp.grad.abs().sum() > 0
assert torch.isfinite(objective_model.vs.grad).all() and objective_model.vs.grad.abs().sum() > 0

epsilons_tensor = torch.tensor(epsilons, dtype=torch.float64)
reference = full_remainders[0] * (epsilons_tensor / epsilons_tensor[0]).square()
figure = plt.figure(figsize=(5, 4))
plt.loglog(epsilons, full_remainders, "o-", label="Full-chain remainder")
plt.loglog(epsilons, reference, "--", label=r"$O(\epsilon^2)$")
plt.xlabel(r"$\epsilon$")
plt.ylabel("Taylor remainder")
plt.title("Global Vp Taylor test")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
figure.savefig(FIGURES / "taylor_remainders.png", dpi=200)
plt.show()

print("test_gradient.py: passed")
