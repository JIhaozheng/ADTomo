from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from adtomo import ForwardGrid, VelocityModel


FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)

lon = torch.arange(-121.0, -118.9, 0.1, dtype=torch.float64)
lat = torch.arange(33.8, 36.1, 0.1, dtype=torch.float64)
depth = torch.arange(-20.0, 55.1, 5.0, dtype=torch.float64)
depth_grid, lat_grid, lon_grid = torch.meshgrid(depth, lat, lon, indexing="ij")
vp = torch.full((len(depth), len(lat), len(lon)), 6.0, dtype=torch.float64)
model = VelocityModel(lon, lat, depth, vp, vp / 1.73, trainable=False)

station_lonlatdepth = torch.tensor([-120.0, 35.0, 0.0], dtype=torch.float64)
event_lonlatdepth = torch.tensor([[-119.9, 35.1, 8.0]], dtype=torch.float64)
grid = ForwardGrid(station_lonlatdepth, event_lonlatdepth, model, spacing=5.0)

# This asymmetric linear field exposes both interpolation and axis-order errors.
global_field = 0.01 * depth_grid + 0.1 * lat_grid + lon_grid
local_field = grid.sample(global_field)
sample_coordinates = grid.model_sample_grid[0]
local_lon = model.lon[0] + (sample_coordinates[..., 0] + 1) * (model.lon[-1] - model.lon[0]) / 2
local_lat = model.lat[0] + (sample_coordinates[..., 1] + 1) * (model.lat[-1] - model.lat[0]) / 2
local_depth = model.depth[0] + (sample_coordinates[..., 2] + 1) * (model.depth[-1] - model.depth[0]) / 2
expected_field = 0.01 * local_depth + 0.1 * local_lat + local_lon
assert torch.allclose(local_field, expected_field, atol=1e-9)

global_depth_index = int(torch.argmin((model.depth - 10.0).abs()))
local_down_index = int(torch.argmin((grid.z - 10.0).abs()))
lon_indices = torch.where((model.lon >= local_lon.min()) & (model.lon <= local_lon.max()))[0]
lat_indices = torch.where((model.lat >= local_lat.min()) & (model.lat <= local_lat.max()))[0]
global_slice = global_field[global_depth_index][lat_indices][:, lon_indices]
local_slice = local_field[local_down_index]
vmin = min(global_slice.min().item(), local_slice.min().item())
vmax = max(global_slice.max().item(), local_slice.max().item())

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
image = plt.imshow(
    global_slice,
    origin="lower",
    extent=[model.lon[lon_indices[0]], model.lon[lon_indices[-1]], model.lat[lat_indices[0]], model.lat[lat_indices[-1]]],
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
)
plt.title(f"Global field at depth {model.depth[global_depth_index].item():.1f} km")
plt.xlabel("Longitude (deg)")
plt.ylabel("Latitude (deg)")
plt.colorbar(image, label="Synthetic field")

plt.subplot(1, 2, 2)
image = plt.imshow(
    local_slice,
    origin="lower",
    extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]],
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
)
plt.title(f"Local field at Down={grid.z[local_down_index].item():.1f} km")
plt.xlabel("East (km)")
plt.ylabel("North (km)")
plt.colorbar(image, label="Synthetic field")
plt.tight_layout()
plt.savefig(FIGURES / "grid_interpolation.png", dpi=200)
plt.close()

print(f"{Path(__file__).name}: passed")
