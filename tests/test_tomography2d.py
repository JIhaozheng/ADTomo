"""Tomography2D checks: 1-D velocity with fixed events, then relocation with fixed velocity."""

import torch

from adtomo import ForwardGrid2D, Tomography2D, VelocityModel1D, predict_travel_times_2d


DEPTH = torch.arange(-5.0, 20.1, 1.0, dtype=torch.float64)
STATIONS = torch.tensor(
    [
        [-122.80, 38.80, 0.0],
        [-122.75, 38.83, 0.0],
        [-122.85, 38.78, 0.0],
        [-122.78, 38.86, 0.0],
        [-122.83, 38.84, 0.0],
    ],
    dtype=torch.float64,
)
EVENTS = torch.tensor([[-122.79, 38.81, 6.0], [-122.81, 38.82, 4.0]], dtype=torch.float64)
EVENT_TIME = torch.tensor([0.0, 10.0], dtype=torch.float64)
SPACING = 0.5


def true_model():
    vp = 5.0 + 0.05 * DEPTH.clamp_min(0.0)
    return VelocityModel1D(DEPTH, vp, vp / 1.73, trainable=False)


def observe(model):
    observed = []
    for station in STATIONS:
        grid = ForwardGrid2D(station, EVENTS, model, spacing=SPACING)
        with torch.no_grad():
            observed.append({phase: EVENT_TIME + predict_travel_times_2d(model, grid, phase, EVENTS) for phase in ("P", "S")})
    return observed


def make_groups(model, initial_loc, observed):
    indices = torch.arange(len(EVENTS))
    return [
        (
            ForwardGrid2D(station, initial_loc, model, spacing=SPACING, padding=4.0),
            [(phase, indices, times) for phase, times in station_observed.items()],
        )
        for station, station_observed in zip(STATIONS, observed)
    ]


def optimize(tomography, groups, iterations, learning_rate, gamma=1.0):
    parameters = [parameter for parameter in tomography.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    initial_loss = tomography(groups).item()
    for _ in range(iterations):
        optimizer.zero_grad()
        loss = tomography(groups)
        loss.backward()
        optimizer.step()
        scheduler.step()
    return initial_loss, tomography(groups).item()


def test_velocity_only_recovers_sampled_depths():
    model = true_model()
    observed = observe(model)
    initial = VelocityModel1D(DEPTH, model.vp.detach() * 1.1, model.vs.detach() * 1.1, trainable=True)
    tomography = Tomography2D(initial, EVENTS, EVENT_TIME, trainable_location=False, trainable_time=False)
    groups = make_groups(initial, EVENTS, observed)

    first, last = optimize(tomography, groups, 200, 0.02)

    assert last < first / 20
    assert torch.isfinite(initial.vp.grad).all() and torch.isfinite(initial.vs.grad).all()
    assert tomography.event_loc.grad is None and tomography.event_time_correction.grad is None
    sampled = (DEPTH >= 0.0) & (DEPTH <= 6.0)
    initial_error = (0.1 * model.vp[sampled]).abs().mean()
    final_error = (initial.vp.detach()[sampled] - model.vp[sampled]).abs().mean()
    assert final_error < initial_error / 2


def optimize_lbfgs(tomography, groups, parameters, rounds=5, max_iter=20):
    optimizer = torch.optim.LBFGS(
        parameters, max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14
    )

    def closure():
        optimizer.zero_grad()
        try:
            loss = tomography(groups)
        except ValueError:  # line-search probe left the forward grid: barrier
            return torch.full((), 1e6, dtype=torch.float64)
        loss.backward()
        return loss

    initial_loss = tomography(groups).item()
    for _ in range(rounds):
        optimizer.step(closure)
    return initial_loss, tomography(groups).item()


def test_location_only_recovers_perturbed_events():
    model = true_model()
    observed = observe(model)
    perturbation = torch.tensor([[0.015, -0.010, -1.5], [-0.010, 0.012, 1.0]], dtype=torch.float64)
    initial_loc = EVENTS + perturbation
    tomography = Tomography2D(model, initial_loc, EVENT_TIME - 0.4)
    groups = make_groups(model, initial_loc, observed)

    first, last = optimize_lbfgs(tomography, groups, [tomography.event_loc, tomography.event_time_correction])

    assert last < first * 1e-6
    assert torch.isfinite(tomography.event_loc.grad).all()
    assert torch.isfinite(tomography.event_time_correction.grad).all()
    assert model.vp.grad is None
    recovered = tomography.event_loc.detach()
    initial_offset = torch.linalg.vector_norm(perturbation[:, :2], dim=-1)
    recovered_offset = torch.linalg.vector_norm(recovered[:, :2] - EVENTS[:, :2], dim=-1)
    assert torch.all(recovered_offset < initial_offset / 100)
    assert torch.all((tomography.event_loc.detach()[:, 2] - EVENTS[:, 2]).abs() < 1e-3)
    assert torch.all((tomography.event_time_correction.detach() - 0.4).abs() < 1e-3)
