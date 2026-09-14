"""Generate true and perturbed initial spherical velocity models."""

from pathlib import Path

import torch

DATA = Path("data")
DATA.mkdir(exist_ok=True)

lon = torch.arange(-120.8, -119.19, 0.1, dtype=torch.float64)
lat = torch.arange(34.2, 35.81, 0.1, dtype=torch.float64)
depth = torch.arange(-15.0, 50.1, 5.0, dtype=torch.float64)
depth_grid, lat_grid, lon_grid = torch.meshgrid(depth, lat, lon, indexing="ij")
vp0 = 5.5 + 0.03 * depth_grid.clamp_min(0.0)
anomaly = 0.35 * torch.exp(
    -((lon_grid + 120.0) / 0.22) ** 2
    - ((lat_grid - 35.0) / 0.20) ** 2
    - ((depth_grid - 15.0) / 10.0) ** 2
)
vp_true = vp0 + anomaly
vs_true = vp_true / 1.73
vp_initial = vp0 + 0.05
vs_initial = vp_initial / 1.73
torch.save({"lon": lon, "lat": lat, "depth": depth, "vp": vp_true, "vs": vs_true}, DATA / "model_true.pt")
torch.save({"lon": lon, "lat": lat, "depth": depth, "vp": vp_initial, "vs": vs_initial}, DATA / "model_initial.pt")
print(f"saved models with shape {tuple(vp_true.shape)} to {DATA}")
