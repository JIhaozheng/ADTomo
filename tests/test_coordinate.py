from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from adtomo.coordinate import ecef_to_local, ecef_to_spherical, local_basis, local_to_ecef, spherical_to_ecef


FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)

station_lonlatdepth = torch.tensor([-120.0, 35.0, 0.0], dtype=torch.float64)
event_lonlatdepth = torch.tensor(
    [[-120.18, 34.90, 8.0], [-119.82, 35.08, 12.0], [-120.06, 35.22, 16.0]], dtype=torch.float64
)
points_lonlatdepth = torch.cat([station_lonlatdepth[None], event_lonlatdepth], dim=0)
points_ecef = spherical_to_ecef(points_lonlatdepth[:, 0], points_lonlatdepth[:, 1], points_lonlatdepth[:, 2])
lon, lat, depth = ecef_to_spherical(points_ecef)
assert torch.allclose(torch.stack([lon, lat, depth], dim=-1), points_lonlatdepth, atol=1e-10)

basis = local_basis(station_lonlatdepth[0], station_lonlatdepth[1])
assert torch.allclose(basis @ basis.T, torch.eye(3, dtype=torch.float64), atol=1e-12)
event_local = ecef_to_local(points_ecef[1:], points_ecef[0], basis)
event_ecef_roundtrip = local_to_ecef(event_local, points_ecef[0], basis)
assert torch.allclose(event_ecef_roundtrip, points_ecef[1:], atol=1e-10)
assert torch.allclose(ecef_to_local(event_ecef_roundtrip, points_ecef[0], basis), event_local, atol=1e-10)

colors = torch.arange(len(event_lonlatdepth))
plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
plt.scatter(event_lonlatdepth[:, 0], event_lonlatdepth[:, 1], c=colors, cmap="viridis", label="events")
plt.plot(station_lonlatdepth[0], station_lonlatdepth[1], "r*", ms=12, label="station")
plt.xlabel("Longitude (deg)")
plt.ylabel("Latitude (deg)")
plt.title("Spherical coordinates")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(event_local[:, 0], event_local[:, 1], c=colors, cmap="viridis", label="events")
plt.plot(0.0, 0.0, "r*", ms=12, label="station")
plt.xlabel("East (km)")
plt.ylabel("North (km)")
plt.title("Station-local coordinates")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "coordinate_transform.png", dpi=200)
plt.close()

print(f"{Path(__file__).name}: passed")
