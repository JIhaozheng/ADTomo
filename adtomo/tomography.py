"""Small functions connecting a velocity model, station grid, and observations."""

from .eikonal3d import solve_eikonal3d


def predict_travel_times(model, grid, phase, event_indices=None, events=None):
    """Travel times for P or S events on one station's cached forward grid."""
    phase = str(phase).upper()
    if phase not in {"P", "S"}:
        raise ValueError("phase must be 'P' or 'S'")
    velocity = grid.sample(model.vp if phase == "P" else model.vs)
    traveltime = solve_eikonal3d(velocity, grid.station_index, grid.spacing)
    return grid.sample_events(traveltime, event_indices=event_indices, events=events)


def predict_phase_times(model, grid, phase, event_dt, event_indices=None, events=None):
    """Predict phase time: selected catalog-event correction plus travel time."""
    travel_time = predict_travel_times(model, grid, phase, event_indices=event_indices, events=events)
    return event_dt.reshape(-1) + travel_time
