"""Small functions connecting a velocity model, station grid, and observations."""

from .eikonal3d import solve_eikonal3d


def predict_travel_times(model, grid, phase, event_indices=None):
    """Travel times for P or S events on one station's cached forward grid."""
    velocity = grid.sample({"P": model.vp, "S": model.vs}[phase.upper()])
    traveltime = solve_eikonal3d(velocity, grid.station_index, grid.spacing)
    return grid.sample_events(traveltime, event_indices=event_indices)
