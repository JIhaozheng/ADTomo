"""Generate absolute P and S phase timestamps from the true model."""

from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adtomo import ForwardGrid, VelocityModel, predict_phase_times

DATA = Path(__file__).resolve().parent / "data"


def load_model(path, trainable=False):
    data = torch.load(path, weights_only=True)
    return VelocityModel(**data, trainable=trainable)


def main():
    model = load_model(DATA / "model_true.pt")
    stations = pd.read_csv(DATA / "stations.csv", dtype={"station_id": str})
    events = pd.read_csv(DATA / "events.csv", dtype={"event_id": str})
    event_xyz = torch.tensor(events[["longitude", "latitude", "depth_km"]].values, dtype=torch.float64)
    picks = []
    with torch.no_grad():
        for _, station in stations.iterrows():
            station_xyz = torch.tensor(
                [station.longitude, station.latitude, station.depth_km], dtype=torch.float64
            )
            grid = ForwardGrid(station_xyz, event_xyz, model, spacing=5.0)
            for phase in ("P", "S"):
                phase_dt = predict_phase_times(model, grid, phase, torch.zeros(len(events), dtype=torch.float64))
                for event, dt in zip(events.itertuples(index=False), phase_dt.tolist()):
                    picks.append(
                        {
                            "event_id": event.event_id,
                            "station_id": station.station_id,
                            "phase_type": phase,
                            "phase_time": (pd.Timestamp(event.event_time) + pd.Timedelta(seconds=dt)).isoformat(
                                timespec="milliseconds"
                            ),
                        }
                    )
    pd.DataFrame(picks).to_csv(DATA / "picks.csv", index=False)
    print(f"saved {len(picks)} P/S phase picks to {DATA}")


if __name__ == "__main__":
    main()
