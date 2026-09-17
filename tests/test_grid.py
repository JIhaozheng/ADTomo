from pathlib import Path
import math

import matplotlib.pyplot as plt
import torch

from adtomo.grid import (
    ForwardGrid,
    RadialForwardGrid,
    VelocityModel,
    VelocityModel1D,
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
coordinate_points_spherical = torch.cat([coordinate_station_spherical[None], coordinate_events_spherical], dim=0)
coordinate_points_ecef = spherical_to_ecef(
    coordinate_points_spherical[:, 0], coordinate_points_spherical[:, 1], coordinate_points_spherical[:, 2]
)
coordinate_lon, coordinate_lat, coordinate_depth = ecef_to_spherical(coordinate_points_ecef)
assert torch.allclose(
    torch.stack([coordinate_lon, coordinate_lat, coordinate_depth], dim=-1), coordinate_points_spherical, atol=1e-10
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
padding = 2.0 * grid.spacing
station_ecef = spherical_to_ecef(*station_spherical)
station_basis = local_basis(station_spherical[0], station_spherical[1])
events_ecef = spherical_to_ecef(
    events_spherical[:, 0], events_spherical[:, 1], events_spherical[:, 2]
)
physical_points = torch.cat(
    [torch.zeros((1, 3), dtype=torch.float64), ecef_to_local(events_ecef, station_ecef, station_basis)], dim=0
)
physical_minimum, physical_maximum = physical_points.amin(dim=0), physical_points.amax(dim=0)
assert grid.z[0].item() == 0.0
assert grid.station_index[2].item() == 0.0
assert torch.all(grid.z >= 0.0)
assert torch.allclose(grid.station_index, grid.station_index.round(), atol=1e-12)
ix, iy, iz = grid.station_index.round().long()
assert grid.x[ix].item() == 0.0 and grid.y[iy].item() == 0.0 and grid.z[iz].item() == 0.0
assert grid.x[0] <= physical_minimum[0] - padding
assert grid.y[0] <= physical_minimum[1] - padding
assert torch.allclose(grid.z[0], physical_minimum[2])
assert grid.x[-1] >= physical_maximum[0] + padding
assert grid.y[-1] >= physical_maximum[1] + padding
assert grid.z[-1] >= physical_maximum[2] + padding
assert 0.0 <= physical_minimum[0] - padding - grid.x[0] < grid.spacing
assert 0.0 <= grid.x[-1] - (physical_maximum[0] + padding) < grid.spacing
assert 0.0 <= physical_minimum[1] - padding - grid.y[0] < grid.spacing
assert 0.0 <= grid.y[-1] - (physical_maximum[1] + padding) < grid.spacing
old_nz = math.ceil(float((physical_maximum[2] - physical_minimum[2] + 2.0 * padding) / grid.spacing)) + 1
assert len(grid.z) < old_nz

shallower_events_spherical = torch.tensor([[-120.0, 35.0, -1.0]], dtype=torch.float64)
shallower_grid = ForwardGrid(station_spherical, shallower_events_spherical, model, spacing=5.0)
assert shallower_grid.z[0] <= -1.0
assert 0.0 <= -1.0 - shallower_grid.z[0] < shallower_grid.spacing
assert shallower_grid.station_index[2].item() > 0.0
assert torch.allclose(shallower_grid.station_index, shallower_grid.station_index.round(), atol=1e-12)
ix, iy, iz = shallower_grid.station_index.round().long()
assert shallower_grid.x[ix].item() == 0.0
assert shallower_grid.y[iy].item() == 0.0
assert shallower_grid.z[iz].item() == 0.0

# This asymmetric linear field exposes both interpolation and axis-order errors.
global_field = 0.01 * depth_grid + 0.1 * lat_grid + lon_grid
local_field = grid.sample_model(global_field)
sample_coordinates = grid.sample_grid[0]
local_lon = model.lon[0] + (sample_coordinates[..., 0] + 1) * (model.lon[-1] - model.lon[0]) / 2
local_lat = model.lat[0] + (sample_coordinates[..., 1] + 1) * (model.lat[-1] - model.lat[0]) / 2
local_depth = model.depth[0] + (sample_coordinates[..., 2] + 1) * (model.depth[-1] - model.depth[0]) / 2
expected_field = 0.01 * local_depth + 0.1 * local_lat + local_lon
assert torch.allclose(local_field, expected_field, atol=1e-9)
assert local_field.shape == (len(grid.z), len(grid.y), len(grid.x))

# Event interpolation accepts the same physical (East, North, Down) layout.
local_coordinate_field = torch.arange(len(grid.z), dtype=torch.float64)[:, None, None]
local_coordinate_field = local_coordinate_field + 10.0 * torch.arange(len(grid.y), dtype=torch.float64)[None, :, None]
local_coordinate_field = local_coordinate_field + 100.0 * torch.arange(len(grid.x), dtype=torch.float64)[None, None, :]
expected_event_value = 100.0 * grid.events_index[:, 0] + 10.0 * grid.events_index[:, 1] + grid.events_index[:, 2]
assert torch.allclose(grid.sample_events(local_coordinate_field), expected_event_value, atol=1e-12)

global_depth_index = int(torch.argmin((model.depth - 10.0).abs()))
local_down_index = int(torch.argmin((grid.z - 10.0).abs()))
lon_indices = torch.where((model.lon >= local_lon.min()) & (model.lon <= local_lon.max()))[0]
lat_indices = torch.where((model.lat >= local_lat.min()) & (model.lat <= local_lat.max()))[0]
global_slice = global_field[global_depth_index][lat_indices][:, lon_indices]
local_slice = local_field[local_down_index]
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

# A depth-only model's 2-D (depth, range) forward grid must match the 3-D
# constant-velocity analytic travel time to the same discretization tolerance
# as the 3-D solver, and index_from_spherical must agree with the cached
# build-time events_index for the same (unperturbed) event positions.
radial_depth = torch.arange(-5.0, 20.1, 1.0, dtype=torch.float64)
radial_velocity_km_s = 5.0
radial_vp = torch.full_like(radial_depth, radial_velocity_km_s)
radial_model = VelocityModel1D(radial_depth, radial_vp, radial_vp / 1.73, trainable=False)
radial_station = torch.tensor([-122.80, 38.80, 0.0], dtype=torch.float64)
radial_events = torch.tensor([[-122.79, 38.81, 5.0], [-122.75, 38.79, 10.0]], dtype=torch.float64)
radial_grid = RadialForwardGrid(radial_station, radial_events, radial_model, spacing=0.5)
assert radial_grid.r[0].item() == 0.0
assert radial_grid.station_index[0].item() == 0.0

radial_station_ecef = spherical_to_ecef(*radial_station)
radial_basis = local_basis(radial_station[0], radial_station[1])
radial_events_ecef = spherical_to_ecef(radial_events[:, 0], radial_events[:, 1], radial_events[:, 2])
radial_events_local = ecef_to_local(radial_events_ecef, radial_station_ecef, radial_basis)
radial_analytic = torch.linalg.vector_norm(radial_events_local, dim=-1) / radial_velocity_km_s

from adtomo import predict_travel_times_2d

radial_traveltime = predict_travel_times_2d(radial_model, radial_grid, "P")
assert torch.isfinite(radial_traveltime).all()
assert (radial_traveltime - radial_analytic).abs().max() < 2.0 * radial_grid.spacing / radial_velocity_km_s

assert torch.allclose(radial_grid.index_from_spherical(radial_events), radial_grid.events_index, atol=1e-10)

print("test_grid.py: passed")
