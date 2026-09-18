"""ForwardGrid2D must be the azimuthal reduction of the 3-D ForwardGrid geometry (Review.md section 10)."""

import torch

from adtomo import ForwardGrid, ForwardGrid2D, VelocityModel, VelocityModel1D, predict_travel_times, predict_travel_times_2d
from adtomo.grid import ecef_to_local, spherical_to_ecef


STATION = torch.tensor([-122.80, 38.80, 0.0], dtype=torch.float64)
EVENTS = torch.tensor([[-122.70, 38.85, 6.0], [-122.86, 38.75, 3.0], [-122.78, 38.83, 9.0]], dtype=torch.float64)
DEPTH = torch.arange(-5.0, 25.1, 1.0, dtype=torch.float64)
VP = 5.0 + 0.05 * DEPTH.clamp_min(0.0)


def models():
    model_1d = VelocityModel1D(DEPTH, VP, VP / 1.73, trainable=False)
    lon = torch.arange(-123.3, -122.29, 0.1, dtype=torch.float64)
    lat = torch.arange(38.3, 39.31, 0.1, dtype=torch.float64)
    vp_3d = VP[:, None, None].expand(-1, len(lat), len(lon)).clone()
    model_3d = VelocityModel(lon, lat, DEPTH, vp_3d, vp_3d / 1.73, trainable=False)
    return model_1d, model_3d


def travel_time_difference(spacing):
    model_1d, model_3d = models()
    grid_2d = ForwardGrid2D(STATION, EVENTS, model_1d, spacing)
    grid_3d = ForwardGrid(STATION, EVENTS, model_3d, spacing)
    with torch.no_grad():
        t_2d = predict_travel_times_2d(model_1d, grid_2d, "P", EVENTS)
        t_3d = predict_travel_times(model_3d, grid_3d, "P", EVENTS)
    return (t_2d - t_3d).abs().max().item(), t_3d.max().item()


def test_to_section_is_the_exact_reduction_of_the_3d_local_frame():
    model_1d, model_3d = models()
    grid_2d = ForwardGrid2D(STATION, EVENTS, model_1d, 1.0)
    grid_3d = ForwardGrid(STATION, EVENTS, model_3d, 1.0)
    local = ecef_to_local(spherical_to_ecef(EVENTS[:, 0], EVENTS[:, 1], EVENTS[:, 2]), grid_3d.station_ecef, grid_3d.basis)
    expected = torch.stack([torch.hypot(local[:, 0], local[:, 1]), local[:, 2]], dim=-1)
    assert torch.allclose(grid_2d.to_section(EVENTS), expected, atol=1e-12)


def test_2d_and_3d_travel_times_agree_and_converge():
    # Both solvers are first-order fast-sweeping schemes, so their disagreement
    # must shrink with the spacing (measured: 5.5%, 3.4%, 2.1% of the longest
    # travel time at 1.0, 0.5, 0.25 km) if the 2-D section is the same geometry.
    differences = []
    for spacing in (1.0, 0.5, 0.25):
        difference, travel_time = travel_time_difference(spacing)
        differences.append(difference)
    assert differences[2] < 0.03 * travel_time
    assert differences[0] > differences[1] > differences[2]


def test_event_leaving_the_grid_is_rejected():
    model_1d, _ = models()
    grid_2d = ForwardGrid2D(STATION, EVENTS, model_1d, 1.0)
    far_event = torch.tensor([[-122.20, 38.80, 6.0]], dtype=torch.float64)
    try:
        predict_travel_times_2d(model_1d, grid_2d, "P", far_event)
    except ValueError as error:
        assert "left the fixed 2-D forward grid" in str(error)
    else:
        raise AssertionError("an event outside the section must be rejected")
