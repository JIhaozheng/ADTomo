"""Focused checks for the public checkerboard example."""

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "examples" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkerboard_models_use_unperturbed_background():
    generation = load_script("00_gen_velocity.py")
    initial, true = generation.build_models()

    assert set(initial) == {"lon", "lat", "depth", "vp", "vs"}
    assert set(true) == set(initial)
    assert torch.equal(initial["vs"], initial["vp"] / 1.73)

    for phase in ("vp", "vs"):
        relative = (true[phase] - initial[phase]) / initial[phase]
        assert torch.allclose(
            relative.abs(), torch.full_like(relative, generation.CHECKER_AMPLITUDE)
        )
        assert relative[0, 0, 0] < 0
        assert relative[0, 0, generation.CHECKER_LON_NODES] > 0
        assert relative[0, generation.CHECKER_LAT_NODES, 0] > 0
        assert relative[generation.CHECKER_DEPTH_NODES, 0, 0] > 0


def test_example_defaults_are_script_relative():
    generation = load_script("00_gen_velocity.py")
    assert generation.DATA == ROOT / "examples" / "data"
    assert generation.FIGURES == ROOT / "examples" / "figures"
