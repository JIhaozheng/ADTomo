"""Generate absolute P and S phase timestamps from the true model."""

from pathlib import Path

import pandas as pd
import torch

from adtomo import ForwardGrid, VelocityModel, predict_travel_times

DATA = Path("data")

model = VelocityModel(**torch.load(DATA / "model_true.pt", weights_only=True), trainable=False)
stations = pd.read_csv(DATA / "stations.csv", dtype={"station_id": str})
events = pd.read_csv(DATA / "events.csv", dtype={"event_id": str})
event_lonlatdepth = torch.tensor(events[["longitude", "latitude", "depth_km"]].values, dtype=torch.float64)
picks = []
with torch.no_grad():
    for _, station in stations.iterrows():
        station_lonlatdepth = torch.tensor(
            [station.longitude, station.latitude, station.depth_km], dtype=torch.float64
        )
        grid = ForwardGrid(station_lonlatdepth, event_lonlatdepth, model, spacing=5.0)
        for phase in ("P", "S"):
            travel_times = predict_travel_times(model, grid, phase)
            for event, travel_time in zip(events.itertuples(index=False), travel_times.tolist()):
                picks.append(
                    {
                        "event_id": event.event_id,
                        "station_id": station.station_id,
                        "phase_type": phase,
                        "phase_time": (pd.Timestamp(event.event_time) + pd.Timedelta(seconds=travel_time)).isoformat(
                            timespec="milliseconds"
                        ),
                    }
                )
pd.DataFrame(picks).to_csv(DATA / "picks.csv", index=False)
print(f"saved {len(picks)} P/S phase picks to {DATA}")
