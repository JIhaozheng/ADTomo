"""Stage 3: jointly invert Vp(z)/Vs(z), event locations, and origin-time corrections.

Both the velocity model and the event parameters start perturbed and are
optimized in one computational graph. The check here is that every parameter
receives a finite gradient and the loss decreases; exact recovery is not
expected because the joint problem contains velocity/location trade-offs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from adtomo import Tomography2D, VelocityModel1D

from run_location1d import perturb
from synthetic1d import run_lbfgs, station_groups, synthetic_arrivals, true_events, true_model


FIGURES = Path(__file__).resolve().parent / "figures"
ROUNDS = 15


def main():
    model = true_model()
    event_loc, event_time = true_events()
    observed = synthetic_arrivals(model, event_loc, event_time)

    initial = VelocityModel1D(model.depth, model.vp.detach() * 1.08, model.vs.detach() * 0.94, trainable=True)
    initial_loc, initial_time = perturb(event_loc, event_time)
    tomography = Tomography2D(initial, initial_loc, initial_time)
    groups = station_groups(initial, initial_loc, observed)
    parameters = [initial.vp, initial.vs, tomography.event_loc, tomography.event_time_correction]
    history = run_lbfgs(tomography, groups, parameters, rounds=ROUNDS)
    assert history[-1] < history[0]

    recovered_loc = tomography.event_loc.detach()
    recovered_time = tomography.event_time.detach()
    sampled = (model.depth >= 0.0) & (model.depth <= event_loc[:, 2].max())
    print(f"Vp mean |error| over sampled depths: initial={(0.08 * model.vp)[sampled].abs().mean():.4f} recovered={(initial.vp.detach() - model.vp)[sampled].abs().mean():.4f} km/s")
    print(f"depth error (km): initial={(initial_loc[:, 2] - event_loc[:, 2]).abs().mean():.3f} recovered={(recovered_loc[:, 2] - event_loc[:, 2]).abs().mean():.3f}")
    print(f"origin-time error (s): initial={(initial_time - event_time).abs().mean():.3f} recovered={(recovered_time - event_time).abs().mean():.3f}")

    FIGURES.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    axes[0].semilogy(history, color="tab:blue")
    axes[0].set(title="Joint inversion", xlabel="L-BFGS round", ylabel="arrival-time MSE (s²)")
    axes[0].grid(alpha=0.3)
    for name, truth, start, end, color in (
        ("Vp", model.vp, model.vp * 1.08, initial.vp, "tab:red"),
        ("Vs", model.vs, model.vs * 0.94, initial.vs, "tab:blue"),
    ):
        axes[1].plot(truth, model.depth, "-", color=color, label=f"{name} true")
        axes[1].plot(start, model.depth, ":", color=color, label=f"{name} initial")
        axes[1].plot(end.detach(), model.depth, "--", color=color, label=f"{name} recovered")
    axes[1].invert_yaxis()
    axes[1].set(title="1-D velocity profiles", xlabel="velocity (km/s)", ylabel="depth (km)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[2].scatter(event_loc[:, 0], event_loc[:, 1], marker="*", s=80, color="k", label="true")
    axes[2].scatter(initial_loc[:, 0], initial_loc[:, 1], marker="x", color="tab:gray", label="initial")
    axes[2].scatter(recovered_loc[:, 0], recovered_loc[:, 1], marker="o", facecolors="none", edgecolors="tab:red", label="recovered")
    axes[2].set(title="Epicenters", xlabel="longitude (deg)", ylabel="latitude (deg)")
    axes[2].legend()
    figure.savefig(FIGURES / "joint1d.png", dpi=180)
    plt.close(figure)
    print(f"saved {FIGURES / 'joint1d.png'}")


if __name__ == "__main__":
    main()
