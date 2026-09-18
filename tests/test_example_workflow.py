"""Keep the public example pipeline direct and stage-specific."""

from pathlib import Path


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_example_scripts_keep_scientific_responsibilities_visible():
    velocity = (EXAMPLES / "00_gen_velocity.py").read_text()
    stations = (EXAMPLES / "01_gen_stations.py").read_text()
    events = (EXAMPLES / "02_gen_events.py").read_text()
    picks = (EXAMPLES / "03_gen_picks.py").read_text()
    inversion_3d = (EXAMPLES / "inversion3d.py").read_text()
    inversion_1d = (EXAMPLES / "inversion1d.py").read_text()
    runner = (EXAMPLES / "run_inversion.sh").read_text()

    assert "--model" in velocity and "--wavelength-lon" in velocity and "torch.cos" in velocity
    assert "--num-stations" in stations and "--lon-min" in stations
    assert "--num-events" in events and "--depth-max" in events and "--location-noise-km" in events and "--time-noise-s" in events
    assert "events_initial.csv" in events
    assert "ForwardGrid" in picks and "predict_travel_times" in picks and "ForwardGrid2D" in picks
    for inversion in (inversion_3d, inversion_1d):
        assert "--trainable" in inversion and "--optimizer" in inversion and "requires_grad_" in inversion
        assert "events_initial.csv" in inversion
    assert "inversion${MODEL}.py" in runner
