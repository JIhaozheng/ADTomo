from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from adtomo import ForwardGrid, VelocityModel


def assert_value_error(message, function):
    try:
        function()
    except ValueError as error:
        assert message in str(error)
    else:
        raise AssertionError("expected ValueError")


FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)

lon = torch.arange(-121.0, -118.9, 0.1, dtype=torch.float64)
lat = torch.arange(33.8, 36.1, 0.1, dtype=torch.float64)
depth = torch.arange(-20.0, 55.1, 5.0, dtype=torch.float64)
depth_grid, lat_grid, lon_grid = torch.meshgrid(depth, lat, lon, indexing="ij")
vp = torch.full((len(depth), len(lat), len(lon)), 6.0, dtype=torch.float64)
model = VelocityModel(lon, lat, depth, vp, vp / 1.73, trainable=True)

station_lonlatdepth = torch.tensor([-120.0, 35.0, 0.0], dtype=torch.float64)
event_lonlatdepth = torch.tensor([[-119.8, 35.2, 10.0], [-120.2, 34.9, 17.0]], dtype=torch.float64)
constant_grid = ForwardGrid(station_lonlatdepth, event_lonlatdepth, model, spacing=5.0)
assert constant_grid.shape == (len(constant_grid.z), len(constant_grid.y), len(constant_grid.x))
assert torch.all(constant_grid.station_index >= 0)
assert torch.all(constant_grid.event_index >= 0)
assert torch.allclose(
    constant_grid.sample(model.vp), torch.full(constant_grid.shape, 6.0, dtype=torch.float64), atol=1e-10
)

global_field = 0.01 * depth_grid + 0.1 * lat_grid + lon_grid
single_event_lonlatdepth = torch.tensor([[-119.9, 35.1, 8.0]], dtype=torch.float64)
grid = ForwardGrid(station_lonlatdepth, single_event_lonlatdepth, model, spacing=5.0)
local_field = grid.sample(global_field)
local_lon = model.lon[0] + (grid.model_grid[0, ..., 0] + 1) * (model.lon[-1] - model.lon[0]) / 2
local_lat = model.lat[0] + (grid.model_grid[0, ..., 1] + 1) * (model.lat[-1] - model.lat[0]) / 2
local_depth = model.depth[0] + (grid.model_grid[0, ..., 2] + 1) * (model.depth[-1] - model.depth[0]) / 2
assert torch.allclose(local_field, 0.01 * local_depth + 0.1 * local_lat + local_lon, atol=1e-9)

traveltime_field = torch.arange(torch.tensor(grid.shape).prod(), dtype=torch.float64).reshape(grid.shape)
assert torch.allclose(
    grid.sample_events(traveltime_field),
    grid.sample_events(traveltime_field, events=single_event_lonlatdepth),
    atol=1e-10,
)

halo_model = VelocityModel(lon, lat, depth, vp, vp / 1.73, trainable=True)
halo_model.lon = halo_model.lon[9:-9]
halo_model.vp = torch.nn.Parameter(halo_model.vp[..., 9:-9])
halo_model.vs = torch.nn.Parameter(halo_model.vs[..., 9:-9])
assert_value_error(
    "forward grid needs",
    lambda: ForwardGrid(station_lonlatdepth, single_event_lonlatdepth, halo_model, spacing=5.0),
)

global_depth_index = int(torch.argmin((model.depth - 10.0).abs()))
local_down_index = int(torch.argmin((grid.z - 10.0).abs()))
lon_indices = torch.where((model.lon >= local_lon.min()) & (model.lon <= local_lon.max()))[0]
lat_indices = torch.where((model.lat >= local_lat.min()) & (model.lat <= local_lat.max()))[0]
global_slice = global_field[global_depth_index][lat_indices][:, lon_indices]
local_slice = local_field[local_down_index]
vmin = min(global_slice.min().item(), local_slice.min().item())
vmax = max(global_slice.max().item(), local_slice.max().item())
figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
image = axes[0].imshow(
    global_slice,
    origin="lower",
    extent=[model.lon[lon_indices[0]], model.lon[lon_indices[-1]], model.lat[lat_indices[0]], model.lat[lat_indices[-1]]],
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
)
axes[0].set(title=f"Global field at depth {model.depth[global_depth_index].item():.1f} km", xlabel="Longitude (deg)", ylabel="Latitude (deg)")
figure.colorbar(image, ax=axes[0], label="Synthetic field")

image = axes[1].imshow(
    local_slice,
    origin="lower",
    extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]],
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
)
axes[1].set(title=f"Local field at Down={grid.z[local_down_index].item():.1f} km", xlabel="East (km)", ylabel="North (km)")
figure.colorbar(image, ax=axes[1], label="Synthetic field")
figure.savefig(FIGURES / "grid_interpolation.png", dpi=200)
plt.close(figure)

print(f"{Path(__file__).name}: passed")
