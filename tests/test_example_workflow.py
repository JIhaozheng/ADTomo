"""Focused checks for the public checkerboard example."""

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "examples" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smooth_checkerboard_models_use_unperturbed_background():
    generation = load_script("00_gen_velocity.py")
    initial, true, checker = generation.build_models()

    assert set(initial) == {"lon", "lat", "depth", "vp", "vs"}
    assert set(true) == set(initial)
    assert torch.equal(initial["vs"], initial["vp"] / 1.73)
    assert checker.min() < 0 < checker.max()
    assert checker.abs().max() <= 1.0
    assert torch.allclose(
        checker,
        generation.smooth_checkerboard(
            initial["lon"],
            initial["lat"],
            initial["depth"],
            generation.WAVELENGTH_LON,
            generation.WAVELENGTH_LAT,
            generation.WAVELENGTH_DEPTH,
        ),
    )

    for phase in ("vp", "vs"):
        relative = (true[phase] - initial[phase]) / initial[phase]
        assert torch.allclose(relative, generation.CHECKER_AMPLITUDE * checker)
        assert ((relative.abs() > 0.1 * generation.CHECKER_AMPLITUDE) & (relative.abs() < 0.9 * generation.CHECKER_AMPLITUDE)).any()


def test_default_random_geometry_is_broad_and_reproducible():
    generation = load_script("00_gen_velocity.py")
    stations_script = load_script("01_gen_stations.py")
    events_script = load_script("02_gen_events.py")
    initial, _, _ = generation.build_models()

    stations = stations_script.generate_stations(
        initial,
        stations_script.DEFAULT_NUM_STATIONS,
        stations_script.DEFAULT_SEED,
        stations_script.DEFAULT_LON_MIN,
        stations_script.DEFAULT_LON_MAX,
        stations_script.DEFAULT_LAT_MIN,
        stations_script.DEFAULT_LAT_MAX,
    )
    assert stations.equals(
        stations_script.generate_stations(
            initial,
            stations_script.DEFAULT_NUM_STATIONS,
            stations_script.DEFAULT_SEED,
            stations_script.DEFAULT_LON_MIN,
            stations_script.DEFAULT_LON_MAX,
            stations_script.DEFAULT_LAT_MIN,
            stations_script.DEFAULT_LAT_MAX,
        )
    )
    lon_min, lon_max = stations_script.DEFAULT_LON_MIN, stations_script.DEFAULT_LON_MAX
    lat_min, lat_max = stations_script.DEFAULT_LAT_MIN, stations_script.DEFAULT_LAT_MAX
    assert stations.longitude.between(lon_min, lon_max).all()
    assert stations.latitude.between(lat_min, lat_max).all()
    assert stations.longitude.max() - stations.longitude.min() > 0.5 * (lon_max - lon_min)
    assert stations.latitude.max() - stations.latitude.min() > 0.5 * (lat_max - lat_min)

    events = events_script.generate_events(
        initial,
        events_script.DEFAULT_NUM_EVENTS,
        events_script.DEFAULT_SEED,
        events_script.DEFAULT_LON_MIN,
        events_script.DEFAULT_LON_MAX,
        events_script.DEFAULT_LAT_MIN,
        events_script.DEFAULT_LAT_MAX,
        events_script.DEFAULT_DEPTH_MIN,
        events_script.DEFAULT_DEPTH_MAX,
    )
    assert events.equals(
        events_script.generate_events(
            initial,
            events_script.DEFAULT_NUM_EVENTS,
            events_script.DEFAULT_SEED,
            events_script.DEFAULT_LON_MIN,
            events_script.DEFAULT_LON_MAX,
            events_script.DEFAULT_LAT_MIN,
            events_script.DEFAULT_LAT_MAX,
            events_script.DEFAULT_DEPTH_MIN,
            events_script.DEFAULT_DEPTH_MAX,
        )
    )
    lon_min, lon_max = events_script.DEFAULT_LON_MIN, events_script.DEFAULT_LON_MAX
    lat_min, lat_max = events_script.DEFAULT_LAT_MIN, events_script.DEFAULT_LAT_MAX
    depth_min, depth_max = events_script.DEFAULT_DEPTH_MIN, events_script.DEFAULT_DEPTH_MAX
    assert events.longitude.between(lon_min, lon_max).all()
    assert events.latitude.between(lat_min, lat_max).all()
    assert events.depth_km.between(depth_min, depth_max).all()
    assert events.longitude.max() - events.longitude.min() > 0.5 * (lon_max - lon_min)
    assert events.latitude.max() - events.latitude.min() > 0.5 * (lat_max - lat_min)
    assert events.depth_km.max() - events.depth_km.min() > 0.5 * (depth_max - depth_min)


def test_acquisition_regions_must_be_strictly_inside_the_model():
    generation = load_script("00_gen_velocity.py")
    stations_script = load_script("01_gen_stations.py")
    events_script = load_script("02_gen_events.py")
    initial, _, _ = generation.build_models()

    with pytest.raises(ValueError, match="strictly inside"):
        stations_script.validate_horizontal_region(
            initial, initial["lon"][0].item(), -118.0, 33.8, 36.1
        )
    with pytest.raises(ValueError, match="strictly inside"):
        events_script.validate_acquisition_region(
            initial, -120.6, -118.0, 33.8, 36.1, 2.0, initial["depth"][-1].item()
        )


def test_example_defaults_are_script_relative():
    generation = load_script("00_gen_velocity.py")
    assert generation.DATA == ROOT / "examples" / "data"
    assert generation.FIGURES == ROOT / "examples" / "figures"
