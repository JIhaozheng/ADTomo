"""Keep the public example pipeline direct and stage-specific."""

from pathlib import Path


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_example_scripts_keep_scientific_responsibilities_visible():
    velocity = (EXAMPLES / "00_gen_velocity.py").read_text()
    stations = (EXAMPLES / "01_gen_stations.py").read_text()
    events = (EXAMPLES / "02_gen_events.py").read_text()
    picks = (EXAMPLES / "03_gen_picks.py").read_text()
    inversion = (EXAMPLES / "inversion.py").read_text()
    runner = (EXAMPLES / "run_inversion.sh").read_text()
    pipeline = (EXAMPLES / "run_pipeline.sh").read_text()

    assert "--wavelength-lon" in velocity and "torch.cos" in velocity and "--model" not in velocity
    assert "--num-stations" in stations and "--lon-min" in stations
    assert "--num-events" in events and "--depth-max" in events and "--location-noise-km" in events and "events_initial.csv" in events
    assert "ForwardGrid" in picks and "predict_travel_times" in picks and "ForwardGrid2D" not in picks
    for name in ("--model", "--trainable", "from_3d", "build_station_groups", "set_trainable", "optimize("):
        assert name in inversion
    assert '--model "$MODEL"' in runner
    assert "00_gen_velocity.py" in pipeline and "run_inversion.sh" in pipeline
