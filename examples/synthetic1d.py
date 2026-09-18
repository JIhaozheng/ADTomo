"""Shared synthetic 1-D setup for run_velocity1d.py, run_location1d.py, and run_joint1d.py."""

import torch

from adtomo import ForwardGrid2D, VelocityModel1D, predict_travel_times_2d


SPACING_KM = 0.5
GRID_PADDING_KM = 4.0
NUM_EVENTS = 20
STATIONS = torch.tensor(
    [
        [-122.90, 38.72, 0.0],
        [-122.72, 38.74, 0.0],
        [-122.95, 38.86, 0.0],
        [-122.70, 38.90, 0.0],
        [-122.83, 38.79, 0.0],
        [-122.77, 38.85, 0.0],
        [-122.88, 38.93, 0.0],
        [-122.66, 38.81, 0.0],
    ],
    dtype=torch.float64,
)


def true_model():
    depth = torch.arange(-2.0, 20.1, 1.0, dtype=torch.float64)
    vp = 4.5 + 0.1 * depth.clamp_min(0.0)
    return VelocityModel1D(depth, vp, vp / 1.75, trainable=False)


def true_events(seed=0):
    """Absolute (lon, lat, depth) and origin times (s) for the synthetic catalog."""
    generator = torch.Generator().manual_seed(seed)
    uniform = lambda: torch.rand(NUM_EVENTS, generator=generator, dtype=torch.float64)
    event_loc = torch.stack([-122.86 + 0.12 * uniform(), 38.76 + 0.12 * uniform(), 2.0 + 8.0 * uniform()], dim=-1)
    event_time = 10.0 * torch.arange(NUM_EVENTS, dtype=torch.float64)
    return event_loc, event_time


def synthetic_arrivals(model, event_loc, event_time):
    """Absolute P and S arrival times at every station for every event."""
    observed = []
    for station in STATIONS:
        grid = ForwardGrid2D(station, event_loc, model, spacing=SPACING_KM)
        with torch.no_grad():
            observed.append({phase: event_time + predict_travel_times_2d(model, grid, phase) for phase in ("P", "S")})
    return observed


def station_groups(model, initial_loc, observed):
    """One Cartesian (y, x) section per station, sized around the initial event positions."""
    indices = torch.arange(len(initial_loc))
    return [
        (
            ForwardGrid2D(station, initial_loc, model, spacing=SPACING_KM, padding=GRID_PADDING_KM),
            [(phase, indices, times) for phase, times in station_observed.items()],
        )
        for station, station_observed in zip(STATIONS, observed)
    ]


def run_adam(tomography, groups, parameter_groups, iterations, gamma=1.0, log_every=50):
    """Minimize the objective; returns the loss history and asserts all trained gradients are finite."""
    optimizer = torch.optim.Adam(parameter_groups)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    history = []
    for iteration in range(iterations):
        optimizer.zero_grad()
        loss = tomography(groups)
        loss.backward()
        for group in parameter_groups:
            for parameter in group["params"]:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        optimizer.step()
        scheduler.step()
        history.append(tomography.data_loss.item())
        if iteration % log_every == 0 or iteration == iterations - 1:
            print(f"iteration {iteration:03d}/{iterations} data={history[-1]:.3e} total={loss.item():.3e}")
    return history


def run_lbfgs(tomography, groups, parameters, rounds=10, max_iter=20):
    """Minimize the objective with L-BFGS (strong-Wolfe line search); returns the loss after each round."""
    optimizer = torch.optim.LBFGS(
        parameters, max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14
    )

    def closure():
        optimizer.zero_grad()
        loss = tomography(groups)
        loss.backward()
        for parameter in parameters:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        return loss

    history = [tomography(groups).item()]
    for round_index in range(rounds):
        optimizer.step(closure)
        history.append(tomography(groups).item())
        print(f"round {round_index:02d}/{rounds} data={tomography.data_loss.item():.3e} total={history[-1]:.3e}")
    return history
