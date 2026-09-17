"""Generate absolute P/S arrival timestamps from the true velocity model."""

from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

from adtomo import ForwardGrid, VelocityModel, predict_travel_times


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FORWARD_SPACING_KM = 5.0


def require_inputs():
    required = ("model_true.pt", "stations.csv", "events.csv")
    missing = [DATA / name for name in required if not (DATA / name).is_file()]
    if missing:
        paths = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing synthetic-data inputs:\n  {paths}\nrun steps 00, 01, and 02 first")


def main():
    started = perf_counter()
    require_inputs()
    model = VelocityModel(**torch.load(DATA / "model_true.pt", weights_only=True), trainable=False)
    stations = pd.read_csv(DATA / "stations.csv", dtype={"station_id": str})
    events = pd.read_csv(DATA / "events.csv", dtype={"event_id": str})
    events_spherical = torch.tensor(
        events[["longitude", "latitude", "depth_km"]].to_numpy(), dtype=torch.float64
    )

    picks = []
    with torch.no_grad():
        for station in stations.itertuples(index=False):
            station_spherical = torch.tensor(
                [station.longitude, station.latitude, station.depth_km], dtype=torch.float64
            )
            grid = ForwardGrid(
                station_spherical,
                events_spherical,
                model,
                spacing=FORWARD_SPACING_KM,
            )
            for phase in ("P", "S"):
                travel_times = predict_travel_times(model, grid, phase)
                for event, travel_time in zip(events.itertuples(index=False), travel_times.tolist()):
                    picks.append(
                        {
                            "event_id": event.event_id,
                            "station_id": station.station_id,
                            "phase_type": phase,
                            "phase_time": (
                                pd.Timestamp(event.event_time) + pd.Timedelta(seconds=travel_time)
                            ).isoformat(timespec="milliseconds"),
                        }
                    )

    path = DATA / "picks.csv"
    pd.DataFrame(picks).to_csv(path, index=False)
    p_count = sum(pick["phase_type"] == "P" for pick in picks)
    s_count = len(picks) - p_count
    expected = 2 * len(stations) * len(events)
    if len(picks) != expected:
        raise AssertionError(f"expected {expected} picks but generated {len(picks)}")
    print(f"saved picks to {path}")
    print(
        f"stations={len(stations)} events={len(events)} P picks={p_count} "
        f"S picks={s_count} total picks={len(picks)} runtime={perf_counter() - started:.2f}s"
    )


if __name__ == "__main__":
    main()
