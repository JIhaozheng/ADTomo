"""Stage 2: relocate events and correct origin times with the velocity model fixed and known.

Arrivals come from known true locations/origin times; the inversion starts from
perturbed locations and origin times and optimizes only the event parameters.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from adtomo import Tomography2D

from synthetic1d import run_lbfgs, station_groups, synthetic_arrivals, true_events, true_model


FIGURES = Path(__file__).resolve().parent / "figures"
ROUNDS = 10


def perturb(event_loc, event_time, seed=1):
    generator = torch.Generator().manual_seed(seed)
    normal = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)
    scale = torch.tensor([0.010, 0.010, 1.0], dtype=torch.float64)  # deg, deg, km
    return event_loc + scale * normal(*event_loc.shape), event_time + 0.3 * normal(len(event_time))


def main():
    model = true_model()
    event_loc, event_time = true_events()
    observed = synthetic_arrivals(model, event_loc, event_time)

    initial_loc, initial_time = perturb(event_loc, event_time)
    tomography = Tomography2D(model, initial_loc, initial_time)
    groups = station_groups(model, initial_loc, observed)
    # Gradients at the starting point: finite for both event parameters, and the
    # horizontal location gradients must be nonzero for this non-symmetric geometry.
    tomography(groups).backward()
    assert torch.isfinite(tomography.event_loc.grad).all()
    assert torch.isfinite(tomography.event_time_correction.grad).all()
    assert (tomography.event_loc.grad[:, :2] != 0).all(), "location gradients vanish: check station geometry"
    assert model.vp.grad is None

    history = run_lbfgs(tomography, groups, [tomography.event_loc, tomography.event_time_correction], rounds=ROUNDS)

    recovered_loc = tomography.event_loc.detach()
    recovered_time = tomography.event_time.detach()
    horizontal_km = lambda loc: 111.0 * torch.hypot((loc[:, 0] - event_loc[:, 0]) * torch.cos(torch.deg2rad(event_loc[:, 1])), loc[:, 1] - event_loc[:, 1])
    print(f"horizontal error (km): initial={horizontal_km(initial_loc).mean():.3f} recovered={horizontal_km(recovered_loc).mean():.3f}")
    print(f"depth error (km): initial={(initial_loc[:, 2] - event_loc[:, 2]).abs().mean():.3f} recovered={(recovered_loc[:, 2] - event_loc[:, 2]).abs().mean():.3f}")
    print(f"origin-time error (s): initial={(initial_time - event_time).abs().mean():.3f} recovered={(recovered_time - event_time).abs().mean():.3f}")

    FIGURES.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    axes[0].semilogy(history, color="tab:blue")
    axes[0].set(title="Location-only inversion", xlabel="L-BFGS round", ylabel="arrival-time MSE (s²)")
    axes[0].grid(alpha=0.3)
    axes[1].scatter(event_loc[:, 0], event_loc[:, 1], marker="*", s=80, color="k", label="true")
    axes[1].scatter(initial_loc[:, 0], initial_loc[:, 1], marker="x", color="tab:gray", label="initial")
    axes[1].scatter(recovered_loc[:, 0], recovered_loc[:, 1], marker="o", facecolors="none", edgecolors="tab:red", label="recovered")
    axes[1].set(title="Epicenters", xlabel="longitude (deg)", ylabel="latitude (deg)")
    axes[1].legend()
    axes[2].plot(initial_loc[:, 2] - event_loc[:, 2], "x", color="tab:gray", label="depth error initial (km)")
    axes[2].plot(recovered_loc[:, 2] - event_loc[:, 2], "o", color="tab:red", label="depth error recovered (km)")
    axes[2].plot(initial_time - event_time, "x", color="tab:olive", label="origin-time error initial (s)")
    axes[2].plot(recovered_time - event_time, "o", color="tab:green", label="origin-time error recovered (s)")
    axes[2].axhline(0.0, color="k", lw=0.8)
    axes[2].set(title="Depth and origin-time errors", xlabel="event index")
    axes[2].legend()
    figure.savefig(FIGURES / "location1d.png", dpi=180)
    plt.close(figure)
    print(f"saved {FIGURES / 'location1d.png'}")


if __name__ == "__main__":
    main()
