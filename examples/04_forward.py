"""Show one station's complete spherical forward calculation."""

from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adtomo import ForwardGrid, VelocityModel, predict_phase_times

DATA = Path(__file__).resolve().parent / "data"


def main():
    data = torch.load(DATA / "model_true.pt", weights_only=True)
    model = VelocityModel(**data, trainable=False)
    stations = pd.read_csv(DATA / "stations.csv", dtype={"station_id": str})
    events = pd.read_csv(DATA / "events.csv", dtype={"event_id": str})
    picks = pd.read_csv(DATA / "picks.csv", dtype={"event_id": str, "station_id": str})
    station = stations.iloc[0]
    event_xyz = torch.tensor(events[["longitude", "latitude", "depth_km"]].values, dtype=torch.float64)
    grid = ForwardGrid(
        torch.tensor([station.longitude, station.latitude, station.depth_km], dtype=torch.float64),
        event_xyz,
        model,
        spacing=5.0,
    )
    event_time = pd.to_datetime(events["event_time"])
    station_picks = picks[(picks.station_id == station.station_id) & (picks.phase_type == "P")].copy()
    observed = (pd.to_datetime(station_picks["phase_time"]).to_numpy() - event_time.to_numpy()).astype("timedelta64[ns]")
    observed = torch.tensor(observed.astype("int64") / 1e9, dtype=torch.float64)
    predicted = predict_phase_times(model, grid, "P", torch.zeros(len(events), dtype=torch.float64))
    print(f"station={station.station_id} local shape (z,y,x)={grid.shape}")
    for event_id, obs, pred in zip(station_picks.event_id[:5], observed[:5], predicted[:5]):
        print(f"{event_id}: observed phase_dt={obs.item():.4f}s predicted={pred.item():.4f}s")


if __name__ == "__main__":
    main()
