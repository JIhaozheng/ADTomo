"""Stage 1: invert a depth-only Vp(z)/Vs(z) from synthetic arrivals with events and stations fixed.

Arrivals come from a known 1-D model through the 2-D eikonal solver; the
inversion starts from a perturbed model and optimizes only the velocity profile.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from adtomo import Tomography2D, VelocityModel1D

from synthetic1d import run_adam, station_groups, synthetic_arrivals, true_events, true_model


FIGURES = Path(__file__).resolve().parent / "figures"
ITERATIONS = 300
LEARNING_RATE = 0.02


def main():
    model = true_model()
    event_loc, event_time = true_events()
    observed = synthetic_arrivals(model, event_loc, event_time)

    initial = VelocityModel1D(model.depth, model.vp.detach() * 1.08, model.vs.detach() * 0.94, trainable=True)
    tomography = Tomography2D(initial, event_loc, event_time, trainable_location=False, trainable_time=False)
    groups = station_groups(initial, event_loc, observed)
    history = run_adam(tomography, groups, [{"params": [initial.vp, initial.vs], "lr": LEARNING_RATE}], ITERATIONS)
    assert tomography.event_loc.grad is None and tomography.event_time_correction.grad is None
    assert history[-1] < history[0]

    sampled = (model.depth >= 0.0) & (model.depth <= event_loc[:, 2].max())
    for name, truth, start, end in (("Vp", model.vp, model.vp * 1.08, initial.vp), ("Vs", model.vs, model.vs * 0.94, initial.vs)):
        print(
            f"{name} mean |error| over sampled depths: initial={(start - truth)[sampled].abs().mean():.4f} "
            f"recovered={(end.detach() - truth)[sampled].abs().mean():.4f} km/s"
        )

    FIGURES.mkdir(exist_ok=True)
    figure, (loss_axis, profile_axis) = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    loss_axis.semilogy(history, color="tab:blue")
    loss_axis.set(title="Velocity-only inversion", xlabel="Adam iteration", ylabel="arrival-time MSE (s²)")
    loss_axis.grid(alpha=0.3)
    for name, truth, start, end, color in (
        ("Vp", model.vp, model.vp * 1.08, initial.vp, "tab:red"),
        ("Vs", model.vs, model.vs * 0.94, initial.vs, "tab:blue"),
    ):
        profile_axis.plot(truth, model.depth, "-", color=color, label=f"{name} true")
        profile_axis.plot(start, model.depth, ":", color=color, label=f"{name} initial")
        profile_axis.plot(end.detach(), model.depth, "--", color=color, label=f"{name} recovered")
    profile_axis.invert_yaxis()
    profile_axis.set(title="1-D velocity profiles", xlabel="velocity (km/s)", ylabel="depth (km)")
    profile_axis.legend()
    profile_axis.grid(alpha=0.3)
    figure.savefig(FIGURES / "velocity1d.png", dpi=180)
    plt.close(figure)
    print(f"saved {FIGURES / 'velocity1d.png'}")


if __name__ == "__main__":
    main()
