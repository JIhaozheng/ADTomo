from pathlib import Path

import matplotlib.pyplot as plt
import torch

from adtomo.grid import (
    ForwardGrid,
    VelocityModel,
    ecef_to_local,
    ecef_to_spherical,
    local_basis,
    local_to_ecef,
    spherical_to_ecef,
)


FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

# Spherical/ECEF and station-local END coordinate checks.
coordinate_station_spherical = torch.tensor([-120.0, 35.0, 0.0], dtype=torch.float64)
coordinate_events_spherical = torch.tensor(
    [[-120.18, 34.90, 8.0], [-119.82, 35.08, 12.0], [-120.06, 35.22, 16.0]], dtype=torch.float64
)
coordinate_points_lonlatdepth = torch.cat([coordinate_station_spherical[None], coordinate_events_spherical], dim=0)
coordinate_points_ecef = spherical_to_ecef(
    coordinate_points_lonlatdepth[:, 0], coordinate_points_lonlatdepth[:, 1], coordinate_points_lonlatdepth[:, 2]
)
coordinate_lon, coordinate_lat, coordinate_depth = ecef_to_spherical(coordinate_points_ecef)
assert torch.allclose(
    torch.stack([coordinate_lon, coordinate_lat, coordinate_depth], dim=-1), coordinate_points_lonlatdepth, atol=1e-10
)

coordinate_basis = local_basis(coordinate_station_spherical[0], coordinate_station_spherical[1])
assert torch.allclose(coordinate_basis @ coordinate_basis.T, torch.eye(3, dtype=torch.float64), atol=1e-12)
coordinate_events_local = ecef_to_local(coordinate_points_ecef[1:], coordinate_points_ecef[0], coordinate_basis)
coordinate_events_ecef_roundtrip = local_to_ecef(coordinate_events_local, coordinate_points_ecef[0], coordinate_basis)
assert torch.allclose(coordinate_events_ecef_roundtrip, coordinate_points_ecef[1:], atol=1e-10)
assert torch.allclose(
    ecef_to_local(coordinate_events_ecef_roundtrip, coordinate_points_ecef[0], coordinate_basis), coordinate_events_local, atol=1e-10
)

lon = torch.arange(-121.0, -118.9, 0.1, dtype=torch.float64)
lat = torch.arange(33.8, 36.1, 0.1, dtype=torch.float64)
depth = torch.arange(-20.0, 55.1, 5.0, dtype=torch.float64)
depth_grid, lat_grid, lon_grid = torch.meshgrid(depth, lat, lon, indexing="ij")
vp = torch.full((len(depth), len(lat), len(lon)), 6.0, dtype=torch.float64)
model = VelocityModel(lon, lat, depth, vp, vp / 1.73, trainable=False)

station_spherical = torch.tensor([-120.0, 35.0, 0.0], dtype=torch.float64)
events_spherical = torch.tensor([[-119.9, 35.1, 8.0]], dtype=torch.float64)
grid = ForwardGrid(station_spherical, events_spherical, model, spacing=5.0)

# This asymmetric linear field exposes both interpolation and axis-order errors.
global_field = 0.01 * depth_grid + 0.1 * lat_grid + lon_grid
local_field = grid.sample_model(global_field)
sample_coordinates = grid.sample_grid[0]
local_lon = model.lon[0] + (sample_coordinates[..., 0] + 1) * (model.lon[-1] - model.lon[0]) / 2
local_lat = model.lat[0] + (sample_coordinates[..., 1] + 1) * (model.lat[-1] - model.lat[0]) / 2
local_depth = model.depth[0] + (sample_coordinates[..., 2] + 1) * (model.depth[-1] - model.depth[0]) / 2
expected_field = 0.01 * local_depth + 0.1 * local_lat + local_lon
assert torch.allclose(local_field, expected_field, atol=1e-9)
assert local_field.shape == (len(grid.x), len(grid.y), len(grid.z))

# Event interpolation accepts the same physical (East, North, Down) layout.
local_coordinate_field = 100.0 * torch.arange(len(grid.x), dtype=torch.float64)[:, None, None]
local_coordinate_field = local_coordinate_field + 10.0 * torch.arange(len(grid.y), dtype=torch.float64)[None, :, None]
local_coordinate_field = local_coordinate_field + torch.arange(len(grid.z), dtype=torch.float64)[None, None, :]
expected_event_value = 100.0 * grid.events_index[:, 0] + 10.0 * grid.events_index[:, 1] + grid.events_index[:, 2]
assert torch.allclose(grid.sample_events(local_coordinate_field), expected_event_value, atol=1e-12)

global_depth_index = int(torch.argmin((model.depth - 10.0).abs()))
local_down_index = int(torch.argmin((grid.z - 10.0).abs()))
lon_indices = torch.where((model.lon >= local_lon.min()) & (model.lon <= local_lon.max()))[0]
lat_indices = torch.where((model.lat >= local_lat.min()) & (model.lat <= local_lat.max()))[0]
global_slice = global_field[global_depth_index][lat_indices][:, lon_indices]
local_slice = local_field[:, :, local_down_index].T
vmin = min(global_slice.min().item(), local_slice.min().item())
vmax = max(global_slice.max().item(), local_slice.max().item())

figure = plt.figure(figsize=(10, 4))
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
figure.savefig(FIGURES / "grid_interpolation.png", dpi=200)
plt.show()

print("test_grid.py: passed")
