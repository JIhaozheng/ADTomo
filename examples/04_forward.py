"""Show one station's explicit spherical forward calculation."""

from pathlib import Path

import pandas as pd
import torch

from adtomo import ForwardGrid, VelocityModel, solve_eikonal3d

DATA = Path("data")

model = VelocityModel(**torch.load(DATA / "model_true.pt", weights_only=True), trainable=False)
stations = pd.read_csv(DATA / "stations.csv", dtype={"station_id": str})
events = pd.read_csv(DATA / "events.csv", dtype={"event_id": str})
picks = pd.read_csv(DATA / "picks.csv", dtype={"event_id": str, "station_id": str})
events_by_id = events.set_index("event_id")
stations_by_id = stations.set_index("station_id")

station_id = picks.loc[picks.phase_type == "P", "station_id"].iloc[0]
station = stations_by_id.loc[station_id]
station_picks = picks[(picks.station_id == station_id) & (picks.phase_type == "P")]
station_event_ids = pd.unique(station_picks.event_id)
station_events = events_by_id.loc[station_event_ids]
station_spherical = torch.tensor([station.longitude, station.latitude, station.depth_km], dtype=torch.float64)
events_spherical = torch.tensor(
    station_events[["longitude", "latitude", "depth_km"]].values, dtype=torch.float64
)
grid = ForwardGrid(station_spherical, events_spherical, model, spacing=5.0)

grid_event_indices = torch.tensor(pd.Index(station_event_ids).get_indexer(station_picks.event_id), dtype=torch.long)
catalog_event_time = pd.to_datetime(station_picks.event_id.map(events_by_id.event_time))
observed_phase_dt = torch.tensor(
    (pd.to_datetime(station_picks.phase_time) - catalog_event_time).dt.total_seconds().to_numpy(), dtype=torch.float64
)

local_velocity = grid.sample_model(model.vp)
traveltime_field = solve_eikonal3d(local_velocity, grid.station_index, grid.spacing)
travel_time = grid.sample_events(traveltime_field, event_indices=grid_event_indices)
predicted_phase_dt = travel_time

print(f"station={station_id} local field shape (East, North, Down)={grid.shape}")
for event_id, observed, propagation, predicted in zip(
    station_picks.event_id[:5], observed_phase_dt[:5], travel_time[:5], predicted_phase_dt[:5]
):
    print(
        f"{event_id}: observed phase_dt={observed.item():.4f}s "
        f"travel_time={propagation.item():.4f}s predicted phase_dt={predicted.item():.4f}s"
    )
