"""Keep the public example scripts direct and stage-specific."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_example_scripts_keep_scientific_responsibilities_visible():
    velocity = (ROOT / "examples" / "00_gen_velocity.py").read_text()
    stations = (ROOT / "examples" / "01_gen_stations.py").read_text()
    events = (ROOT / "examples" / "02_gen_events.py").read_text()
    picks = (ROOT / "examples" / "03_gen_picks.py").read_text()
    inversion = (ROOT / "examples" / "inversion.py").read_text()

    assert "--wavelength-lon" in velocity and "torch.cos" in velocity
    assert "--num-stations" in stations and "--lon-min" in stations
    assert "--num-events" in events and "--depth-max" in events
    assert "ForwardGrid" in picks and "predict_travel_times" in picks
    for name in ("def squared_residual_sum", "def regularization_loss", "class DistributedObjective", "def validate_ddp"):
        assert name not in inversion
