from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from adtomo import solve_eikonal2d, solve_eikonal3d


def assert_value_error(message, function):
    try:
        function()
    except ValueError as error:
        assert message in str(error)
    else:
        raise AssertionError("expected ValueError")


FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)

velocity_yx = torch.full((41, 51), 5.0, dtype=torch.float64)
source_xy = (25.2, 20.3)
traveltime_2d = solve_eikonal2d(velocity_yx, source_xy, 1.0)
assert traveltime_2d.shape == velocity_yx.shape
assert torch.isfinite(traveltime_2d).all()

velocity_zyx = torch.full((21, 31, 41), 5.0, dtype=torch.float64)
source_xyz = (20.2, 15.3, 10.1)
traveltime_3d = solve_eikonal3d(velocity_zyx, source_xyz, 1.0)
assert traveltime_3d.shape == velocity_zyx.shape
assert torch.isfinite(traveltime_3d).all()

assert_value_error("CPU float64", lambda: solve_eikonal3d(torch.ones((3, 3, 3)), (1, 1, 1), 1.0))
assert_value_error(
    "field", lambda: solve_eikonal3d(torch.ones((3, 3), dtype=torch.float64), (1, 1, 1), 1.0)
)
assert_value_error(
    "positive", lambda: solve_eikonal2d(torch.zeros((3, 3), dtype=torch.float64), (1, 1), 1.0)
)
assert_value_error(
    "spacing", lambda: solve_eikonal3d(torch.ones((3, 3, 3), dtype=torch.float64), (1, 1, 1), 0.0)
)
assert_value_error(
    "source", lambda: solve_eikonal3d(torch.ones((3, 3, 3), dtype=torch.float64), (2, 1, 1), 1.0)
)

figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
image = axes[0].imshow(traveltime_2d, origin="lower", extent=[0, 50, 0, 40], cmap="viridis")
axes[0].plot(source_xy[0], source_xy[1], "r*", ms=9, label="source")
axes[0].set(title="2-D travel time", xlabel="East (km)", ylabel="North (km)")
axes[0].legend(loc="upper left")
figure.colorbar(image, ax=axes[0], label="Travel time (s)")

source_z_index = round(source_xyz[2])
image = axes[1].imshow(
    traveltime_3d[source_z_index], origin="lower", extent=[0, 40, 0, 30], cmap="viridis"
)
axes[1].plot(source_xyz[0], source_xyz[1], "r*", ms=9, label="source")
axes[1].set(
    title=f"3-D travel time at Down={source_z_index} km",
    xlabel="East (km)",
    ylabel="North (km)",
)
axes[1].legend(loc="upper left")
figure.colorbar(image, ax=axes[1], label="Travel time (s)")
figure.savefig(FIGURES / "eikonal.png", dpi=200)
plt.close(figure)

print(f"{Path(__file__).name}: passed")
