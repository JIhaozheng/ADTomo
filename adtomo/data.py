"""Turn station, event, and pick tables into the station groups that tomography consumes."""

import numpy as np
import pandas as pd
import torch

from .grid import ForwardGrid, ForwardGrid2D


def build_station_groups(stations, events, picks, model, dimension, spacing, padding=None, padding_above=0.0, rank=0, world_size=1):
    """``[(grid, [(phase, event_indices, observed_phase_dt), ...]), ...]`` for this rank's stations.

    ``events`` is the initial catalog; ``event_indices`` index its rows and
    ``observed_phase_dt = phase_time - event_time`` in seconds. Stations are
    dealt round-robin across ranks. ``dimension`` selects
    :class:`~adtomo.grid.ForwardGrid2D` (``"1d"``) or :class:`~adtomo.grid.ForwardGrid` (``"3d"``).
    """
    Grid = {"1d": ForwardGrid2D, "3d": ForwardGrid}[dimension]
    event_index = pd.Index(events.event_id)
    origin_time = pd.to_datetime(events.event_time, format="ISO8601").to_numpy()
    events_spherical = torch.tensor(events[["longitude", "latitude", "depth_km"]].to_numpy(), dtype=torch.float64)
    stations_by_id = stations.set_index("station_id")
    groups = []
    for station_id in list(pd.unique(picks.station_id))[rank::world_size]:
        station = stations_by_id.loc[station_id]
        station_picks = picks[picks.station_id == station_id]
        station_events = events_spherical[event_index.get_indexer(pd.unique(station_picks.event_id))]
        grid = Grid([station.longitude, station.latitude, station.depth_km], station_events, model, spacing, padding, padding_above)
        phase_groups = []
        for phase, phase_picks in station_picks.groupby("phase_type", sort=False):
            indices = event_index.get_indexer(phase_picks.event_id)
            phase_dt = (pd.to_datetime(phase_picks.phase_time, format="ISO8601").to_numpy() - origin_time[indices]) / np.timedelta64(1, "s")
            phase_groups.append((phase, torch.tensor(indices), torch.tensor(phase_dt, dtype=torch.float64)))
        groups.append((grid, phase_groups))
    return groups
