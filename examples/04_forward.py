"""Show one station's explicit spherical forward calculation."""

from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adtomo import ForwardGrid, VelocityModel, solve_eikonal3d

DATA = Path(__file__).resolve().parent / "data"


def require_event_catalog(events):
    if events.event_id.duplicated().any():
        raise ValueError("event catalog event_id values must be unique")
    return events.set_index("event_id", verify_integrity=True)


def main():
    data = torch.load(DATA / "model_true.pt", weights_only=True)
    model = VelocityModel(**data, trainable=False)
    stations = pd.read_csv(DATA / "stations.csv", dtype={"station_id": str})
    events = pd.read_csv(DATA / "events.csv", dtype={"event_id": str})
    picks = pd.read_csv(DATA / "picks.csv", dtype={"event_id": str, "station_id": str})
    events_by_id = require_event_catalog(events)
    catalog_event_index = {event_id: index for index, event_id in enumerate(events.event_id)}

    station = stations.iloc[0]
    station_picks = picks[(picks.station_id == station.station_id) & (picks.phase_type == "P")].copy()
    if station_picks.empty:
        raise ValueError(f"no P picks for station {station.station_id}")
    unknown_event_ids = sorted(set(station_picks.event_id) - set(events_by_id.index))
    if unknown_event_ids:
        raise ValueError(f"picks reference unknown event_id values: {unknown_event_ids}")

    station_event_ids = list(pd.unique(station_picks.event_id))
    station_events = events_by_id.loc[station_event_ids].reset_index()
    station_lonlatdepth = torch.tensor(
        [station.longitude, station.latitude, station.depth_km], dtype=torch.float64
    )
    event_lonlatdepth = torch.tensor(
        station_events[["longitude", "latitude", "depth_km"]].values, dtype=torch.float64
    )
    grid = ForwardGrid(station_lonlatdepth, event_lonlatdepth, model, spacing=5.0)

    grid_event_index = {event_id: index for index, event_id in enumerate(station_event_ids)}
    catalog_event_indices = torch.tensor(
        [catalog_event_index[event_id] for event_id in station_picks.event_id], dtype=torch.long
    )
    grid_event_indices = torch.tensor(
        [grid_event_index[event_id] for event_id in station_picks.event_id], dtype=torch.long
    )
    catalog_event_time = pd.to_datetime(station_picks.event_id.map(events_by_id.event_time))
    observed_phase_dt = torch.tensor(
        (pd.to_datetime(station_picks.phase_time) - catalog_event_time).dt.total_seconds().to_numpy(),
        dtype=torch.float64,
    )

    local_velocity = grid.sample(model.vp)
    traveltime_field = solve_eikonal3d(local_velocity, grid.station_index, grid.spacing)
    travel_time = grid.sample_events(traveltime_field, event_indices=grid_event_indices)
    event_dt = torch.zeros(len(events), dtype=torch.float64)
    pick_event_dt = event_dt[catalog_event_indices]
    predicted_phase_dt = pick_event_dt + travel_time

    print(f"station={station.station_id} local field shape (Down, North, East)={grid.shape}")
    for event_id, observed, propagation, correction, predicted in zip(
        station_picks.event_id[:5],
        observed_phase_dt[:5],
        travel_time[:5],
        pick_event_dt[:5],
        predicted_phase_dt[:5],
    ):
        print(
            f"{event_id}: observed phase_dt={observed.item():.4f}s "
            f"travel_time={propagation.item():.4f}s event_dt={correction.item():.4f}s "
            f"predicted phase_dt={predicted.item():.4f}s"
        )


if __name__ == "__main__":
    main()
