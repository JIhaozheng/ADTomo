from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from adtomo import ForwardGrid, VelocityModel


def make_model():
    lon = torch.arange(-121.0, -118.9, 0.1, dtype=torch.float64)
    lat = torch.arange(33.8, 36.1, 0.1, dtype=torch.float64)
    depth = torch.arange(-20.0, 55.1, 5.0, dtype=torch.float64)
    z, y, x = torch.meshgrid(depth, lat, lon, indexing="ij")
    vp = 6.0 + 0.0 * z
    return VelocityModel(lon, lat, depth, vp, vp / 1.73, trainable=True)


def test_grid_covers_points_and_constant_field():
    model = make_model()
    station = torch.tensor([-120.0, 35.0, 0.0], dtype=torch.float64)
    events = torch.tensor([[-119.8, 35.2, 10.0], [-120.2, 34.9, 17.0]], dtype=torch.float64)
    grid = ForwardGrid(station, events, model, spacing=5.0)
    assert grid.shape == (len(grid.z), len(grid.y), len(grid.x))
    assert torch.all(grid.station_index >= 0)
    assert torch.all(grid.event_index >= 0)
    assert torch.allclose(grid.sample(model.vp), torch.full(grid.shape, 6.0, dtype=torch.float64), atol=1e-10)


def test_linear_interpolation_and_live_events():
    model = make_model()
    z, y, x = torch.meshgrid(model.depth, model.lat, model.lon, indexing="ij")
    field = 0.01 * z + 0.1 * y + x
    station = torch.tensor([-120.0, 35.0, 0.0], dtype=torch.float64)
    events = torch.tensor([[-119.9, 35.1, 8.0]], dtype=torch.float64)
    grid = ForwardGrid(station, events, model, spacing=5.0)
    sampled = grid.sample(field)
    lon = model.lon[0] + (grid.model_grid[0, ..., 0] + 1) * (model.lon[-1] - model.lon[0]) / 2
    lat = model.lat[0] + (grid.model_grid[0, ..., 1] + 1) * (model.lat[-1] - model.lat[0]) / 2
    depth = model.depth[0] + (grid.model_grid[0, ..., 2] + 1) * (model.depth[-1] - model.depth[0]) / 2
    assert torch.allclose(sampled, 0.01 * depth + 0.1 * lat + lon, atol=1e-9)
    time = torch.arange(torch.tensor(grid.shape).prod(), dtype=torch.float64).reshape(grid.shape)
    assert torch.allclose(grid.sample_events(time), grid.sample_events(time, events=events), atol=1e-10)


def test_grid_requires_model_halo():
    model = make_model()
    model.lon = model.lon[9:-9]
    model.vp = torch.nn.Parameter(model.vp[..., 9:-9])
    model.vs = torch.nn.Parameter(model.vs[..., 9:-9])
    with pytest.raises(ValueError, match="forward grid needs"):
        ForwardGrid(
            torch.tensor([-120.0, 35.0, 0.0], dtype=torch.float64),
            torch.tensor([[-119.8, 35.2, 10.0]], dtype=torch.float64),
            model,
            spacing=5.0,
        )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
