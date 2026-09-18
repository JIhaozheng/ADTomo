"""Forward model synthetic P and S picks from the true velocity model and true events.

The forward engine follows the model in ``model_true.pt``: a 3-D lon/lat/depth
grid uses ``ForwardGrid`` + ``predict_travel_times``; a 1-D depth profile uses
``ForwardGrid2D`` + ``predict_travel_times_2d``.
"""

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

from adtomo import ForwardGrid, ForwardGrid2D, VelocityModel, VelocityModel1D, predict_travel_times, predict_travel_times_2d


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing", type=float, default=2.0, help="forward-grid spacing in km")
    args = parser.parse_args()
    if args.spacing <= 0:
        raise ValueError("spacing must be positive")

    required = [DATA / name for name in ("model_true.pt", "stations.csv", "events.csv")]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("run 00_gen_velocity.py, 01_gen_stations.py, and 02_gen_events.py first")
    started = perf_counter()
    true = torch.load(DATA / "model_true.pt", weights_only=True)
    three_dimensional = "lon" in true
    model = (VelocityModel if three_dimensional else VelocityModel1D)(**true, trainable=False)
    stations = pd.read_csv(DATA / "stations.csv", dtype={"station_id": str})
    events = pd.read_csv(DATA / "events.csv", dtype={"event_id": str})
    events_spherical = torch.tensor(events[["longitude", "latitude", "depth_km"]].to_numpy(), dtype=torch.float64)

    picks = []
    with torch.no_grad():
        for station in stations.itertuples(index=False):
            station_spherical = torch.tensor([station.longitude, station.latitude, station.depth_km], dtype=torch.float64)
            try:
                if three_dimensional:
                    grid = ForwardGrid(station_spherical, events_spherical, model, spacing=args.spacing)
                else:
                    grid = ForwardGrid2D(station_spherical, events_spherical, model, spacing=args.spacing)
            except ValueError as error:
                raise ValueError(
                    f"forward-grid model coverage failed for station {station.station_id!r}: {error}\n"
                    "Reduce the acquisition region or enlarge the velocity model."
                ) from error
            for phase in ("P", "S"):
                if three_dimensional:
                    travel_times = predict_travel_times(model, grid, phase)
                else:
                    travel_times = predict_travel_times_2d(model, grid, phase, events_spherical)
                for event, travel_time in zip(events.itertuples(index=False), travel_times.tolist()):
                    picks.append({
                        "event_id": event.event_id,
                        "station_id": station.station_id,
                        "phase_type": phase,
                        "phase_time": (pd.Timestamp(event.event_time) + pd.Timedelta(seconds=travel_time)).isoformat(timespec="milliseconds"),
                    })

    picks = pd.DataFrame(picks)
    picks.to_csv(DATA / "picks.csv", index=False)
    p_count = (picks.phase_type == "P").sum()
    print(f"saved {len(picks)} picks to {DATA / 'picks.csv'} ({'3-D' if three_dimensional else '1-D'} forward model)")
    print(f"stations={len(stations)} events={len(events)} P={p_count} S={len(picks) - p_count} runtime={perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
