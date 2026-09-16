from pathlib import Path
import math

import matplotlib.pyplot as plt
import torch

from adtomo import solve_eikonal2d, solve_eikonal3d


FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

velocity_km_s = 5.0
spacing_km = 1.0

velocity_yx = torch.full((41, 51), velocity_km_s, dtype=torch.float64)
source_xy = torch.tensor([25.2, 20.3], dtype=torch.float64)
traveltime_2d = solve_eikonal2d(velocity_yx, source_xy, spacing_km)
y_local, x_local = torch.meshgrid(
    torch.arange(velocity_yx.shape[0], dtype=torch.float64) * spacing_km,
    torch.arange(velocity_yx.shape[1], dtype=torch.float64) * spacing_km,
    indexing="ij",
)
analytic_2d = torch.hypot(x_local - source_xy[0] * spacing_km, y_local - source_xy[1] * spacing_km) / velocity_km_s
difference_2d = traveltime_2d - analytic_2d
assert torch.isfinite(traveltime_2d).all()
assert difference_2d.abs().max() < 1.5 * spacing_km / velocity_km_s

# Regression for the 2-D source-cell adjoint: both aligned and fractional
# sources must retain a second-order directional Taylor remainder.
for source in ((25.0, 20.0), (25.2, 20.3), (25.1, 20.15)):
    taylor_velocity = torch.full_like(velocity_yx, velocity_km_s, requires_grad=True)
    direction = torch.linspace(-0.02, 0.02, taylor_velocity.numel(), dtype=torch.float64).reshape_as(taylor_velocity)
    objective = solve_eikonal2d(taylor_velocity, torch.tensor(source, dtype=torch.float64), spacing_km)[35, 40]
    objective.backward()
    derivative = (taylor_velocity.grad * direction).sum().item()
    remainders = []
    for epsilon in (1e-2, 1e-3):
        perturbed = solve_eikonal2d(
            taylor_velocity.detach() + epsilon * direction,
            torch.tensor(source, dtype=torch.float64),
            spacing_km,
        )[35, 40]
        remainders.append(abs(perturbed.item() - objective.item() - epsilon * derivative))
    slope = math.log(remainders[1] / remainders[0]) / math.log(0.1)
    assert 1.8 < slope < 2.2

velocity_dne = torch.full((21, 31, 41), velocity_km_s, dtype=torch.float64)
source_xyz = torch.tensor([20.2, 15.3, 10.1], dtype=torch.float64)
traveltime_3d = solve_eikonal3d(velocity_dne, source_xyz, spacing_km)
z_local, y_local, x_local = torch.meshgrid(
    torch.arange(velocity_dne.shape[0], dtype=torch.float64) * spacing_km,
    torch.arange(velocity_dne.shape[1], dtype=torch.float64) * spacing_km,
    torch.arange(velocity_dne.shape[2], dtype=torch.float64) * spacing_km,
    indexing="ij",
)
analytic_3d = torch.sqrt(
    (x_local - source_xyz[0] * spacing_km).square()
    + (y_local - source_xyz[1] * spacing_km).square()
    + (z_local - source_xyz[2] * spacing_km).square()
) / velocity_km_s
difference_3d = traveltime_3d - analytic_3d
assert torch.isfinite(traveltime_3d).all()
assert difference_3d.abs().max() < 2.0 * spacing_km / velocity_km_s

source_z_index = round(source_xyz[2].item())
rows = [
    (traveltime_2d, analytic_2d, difference_2d, source_xy, "2-D"),
    (
        traveltime_3d[source_z_index],
        analytic_3d[source_z_index],
        difference_3d[source_z_index],
        source_xyz[:2],
        f"3-D at Down={source_z_index} km",
    ),
]
figure = plt.figure(figsize=(12, 7))
for row, (numerical, analytic, difference, source, title) in enumerate(rows):
    vmax = max(numerical.max().item(), analytic.max().item())
    difference_limit = difference.abs().max().item()
    for column, (field, label, cmap, limits) in enumerate(
        (
            (numerical, "Numerical", "viridis", (0.0, vmax)),
            (analytic, "Analytic", "viridis", (0.0, vmax)),
            (difference, "Difference", "seismic", (-difference_limit, difference_limit)),
        )
    ):
        plt.subplot(2, 3, row * 3 + column + 1)
        image = plt.imshow(
            field,
            origin="lower",
            extent=[0, field.shape[1] - 1, 0, field.shape[0] - 1],
            cmap=cmap,
            vmin=limits[0],
            vmax=limits[1],
        )
        if column < 2:
            plt.plot(source[0], source[1], "r*", ms=9)
        plt.title(f"{title}: {label}")
        plt.xlabel("East (km)")
        plt.ylabel("North (km)")
        plt.colorbar(image, label="Travel time (s)" if column < 2 else "Difference (s)")
plt.tight_layout()
figure.savefig(FIGURES / "eikonal.png", dpi=200)
plt.show()

print("test_eikonal.py: passed")
