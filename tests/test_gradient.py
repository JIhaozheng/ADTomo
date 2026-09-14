import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from adtomo import ForwardGrid, VelocityModel, predict_phase_times, solve_eikonal3d


FIGURES = Path(__file__).resolve().parent / "figures"
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

# A concise S-wave backward pass proves Vs is connected to prediction.
s_model = VelocityModel(lon, lat, depth, vp, vp / 1.73, trainable=True)
s_grid = ForwardGrid(station_lonlatdepth, event_lonlatdepth, s_model, spacing=5.0)
s_loss = (predict_phase_times(s_model, s_grid, "S", torch.zeros(1, dtype=torch.float64)) - 5.0).square().sum()
s_loss.backward()
assert s_model.vs.grad is not None
assert torch.isfinite(s_model.vs.grad).all()
assert s_model.vs.grad.abs().sum() > 0

# Full global-Vp chain: sampling, solver, event-level correction selection, loss.
taylor_model = VelocityModel(lon, lat, depth, vp, vp / 1.73, trainable=True)
catalog_event_lonlatdepth = torch.tensor([[-119.9, 35.1, 10.0], [-120.1, 34.9, 12.0]], dtype=torch.float64)
taylor_grid = ForwardGrid(station_lonlatdepth, catalog_event_lonlatdepth[1:], taylor_model, spacing=5.0)
catalog_event_indices = torch.tensor([1], dtype=torch.long)
grid_event_indices = torch.tensor([0], dtype=torch.long)
event_dt = torch.tensor([-0.10, 0.05], dtype=torch.float64)
global_direction = torch.linspace(-0.01, 0.01, taylor_model.vp.numel(), dtype=torch.float64).reshape_as(taylor_model.vp)


def full_phase_time_loss(candidate):
    predicted_phase_dt = predict_phase_times(
        candidate,
        taylor_grid,
        "P",
        event_dt[catalog_event_indices],
        event_indices=grid_event_indices,
    )
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

epsilons_tensor = torch.tensor(epsilons, dtype=torch.float64)
reference = full_remainders[0] * (epsilons_tensor / epsilons_tensor[0]).square()
plt.figure(figsize=(5, 4))
plt.loglog(epsilons, full_remainders, "o-", label="Full-chain remainder")
plt.loglog(epsilons, reference, "--", label=r"$O(\epsilon^2)$")
plt.xlabel(r"$\epsilon$")
plt.ylabel("Taylor remainder")
plt.title("Global Vp Taylor test")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "taylor_remainders.png", dpi=200)
plt.close()

print(f"{Path(__file__).name}: passed")
