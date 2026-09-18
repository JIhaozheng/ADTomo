"""Invert the synthetic 3-D dataset: Vp/Vs and/or event locations and origin times.

Choose what is trainable, as in AI4EPS/ADTomo (parameters are toggled with
``requires_grad``):

    python inversion3d.py --trainable vp,vs
    python inversion3d.py --trainable event_loc,event_time
    python inversion3d.py --trainable vp,vs,event_loc,event_time
    torchrun --standalone --nproc_per_node=4 inversion3d.py --trainable vp,vs

Under ``torchrun`` every rank owns a disjoint station subset; gradients (and
the loss) are summed across ranks inside the optimizer closure, which is
correct for both the dense velocity field and the sparse per-event parameters.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from adtomo import ForwardGrid, Tomography, VelocityModel


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
TRAINABLE = ("vp", "vs", "event_loc", "event_time")
KM_PER_DEGREE = 111.19


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trainable", default="vp,vs", help=f"comma-separated subset of {','.join(TRAINABLE)}")
    parser.add_argument("--optimizer", choices=("lbfgs", "adam"), default="lbfgs")
    parser.add_argument("--iterations", type=int, default=30, help="L-BFGS steps (20 inner iterations each) or Adam iterations")
    parser.add_argument("--learning-rate", type=float, default=None, help="default 1.0 for lbfgs, 0.03 for adam")
    parser.add_argument("--spacing", type=float, default=4.0, help="forward-grid spacing in km")
    parser.add_argument("--lambda-vp", type=float, default=0.0)
    parser.add_argument("--lambda-vs", type=float, default=0.0)
    parser.add_argument("--alpha-vp", type=float, default=0.0)
    parser.add_argument("--alpha-vs", type=float, default=0.0)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    args.trainable = [name for name in args.trainable.split(",") if name]
    unknown = set(args.trainable) - set(TRAINABLE)
    if unknown or not args.trainable:
        raise ValueError(f"--trainable must be a non-empty subset of {TRAINABLE}, got {args.trainable}")
    if args.iterations < 1 or args.spacing <= 0:
        raise ValueError("iterations and spacing must be positive")
    if args.learning_rate is None:
        args.learning_rate = 1.0 if args.optimizer == "lbfgs" else 0.03
    for name in ("data_dir", "results_dir", "figures_dir"):
        path = getattr(args, name)
        setattr(args, name, path if path.is_absolute() else ROOT / path)
    return args


def distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return 0, 1
    dist.init_process_group("gloo")
    return dist.get_rank(), world_size


def load_data(data_dir):
    required = [data_dir / name for name in ("model_initial.pt", "model_true.pt", "stations.csv", "events.csv", "picks.csv")]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("run 00_gen_velocity.py through 03_gen_picks.py first")
    initial = torch.load(data_dir / "model_initial.pt", weights_only=True)
    true = torch.load(data_dir / "model_true.pt", weights_only=True)
    stations = pd.read_csv(data_dir / "stations.csv", dtype={"station_id": str})
    events = pd.read_csv(data_dir / "events.csv", dtype={"event_id": str})
    initial_path = data_dir / "events_initial.csv"
    events_initial = pd.read_csv(initial_path, dtype={"event_id": str}) if initial_path.is_file() else events.copy()
    picks = pd.read_csv(data_dir / "picks.csv", dtype={"event_id": str, "station_id": str})
    return initial, true, stations, events, events_initial, picks


def prepare_station_groups(stations, events_initial, picks, model, spacing, rank, world_size):
    """One forward grid per station (this rank's share), phase groups indexing the global initial catalog."""
    event_index = pd.Index(events_initial.event_id)
    origin_time = pd.to_datetime(events_initial.event_time).to_numpy()
    events_spherical = torch.tensor(events_initial[["longitude", "latitude", "depth_km"]].to_numpy(), dtype=torch.float64)
    stations_by_id = stations.set_index("station_id")
    groups = []
    for station_id in list(pd.unique(picks.station_id))[rank::world_size]:
        station = stations_by_id.loc[station_id]
        station_picks = picks[picks.station_id == station_id]
        station_events = event_index.get_indexer(pd.unique(station_picks.event_id))
        grid = ForwardGrid(
            torch.tensor([station.longitude, station.latitude, station.depth_km], dtype=torch.float64),
            events_spherical[station_events],
            model,
            spacing=spacing,
        )
        phase_groups = []
        for phase, phase_picks in station_picks.groupby("phase_type", sort=False):
            indices = event_index.get_indexer(phase_picks.event_id)
            phase_dt = (pd.to_datetime(phase_picks.phase_time).to_numpy() - origin_time[indices]) / np.timedelta64(1, "s")
            phase_groups.append((phase, torch.tensor(indices, dtype=torch.long), torch.tensor(phase_dt, dtype=torch.float64)))
        groups.append((grid, phase_groups))
    return groups


def configure_trainable(tomography, trainable):
    tomography.model.vp.requires_grad_("vp" in trainable)
    tomography.model.vs.requires_grad_("vs" in trainable)
    tomography.event_loc.requires_grad_("event_loc" in trainable)
    tomography.event_time_correction.requires_grad_("event_time" in trainable)
    return [parameter for parameter in tomography.parameters() if parameter.requires_grad]


def optimize(tomography, groups, parameters, args, total_observations, rank, world_size):
    """Run L-BFGS or Adam; gradients and loss are summed over ranks inside the closure."""
    if args.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(
            parameters, lr=args.learning_rate, max_iter=20, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14
        )
    else:
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    def objective():
        return tomography(groups, data_scale=1.0 / total_observations, regularization_scale=1.0 / world_size)

    def closure():
        optimizer.zero_grad()
        failed = torch.zeros((), dtype=torch.float64)
        try:
            loss = objective()
            loss.backward()
        except ValueError:  # a line-search probe moved an event outside its fixed forward grid
            failed += 1.0
        if world_size > 1:
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            for parameter in parameters:  # keep the optimizer state identical on every rank
                parameter.grad = torch.zeros_like(parameter)
            return torch.full((), 1e6, dtype=torch.float64)
        loss = loss.detach().clone()
        if world_size > 1:
            for parameter in parameters:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        return loss

    def data_loss():
        with torch.no_grad():
            objective()
        data_sum = tomography.data_sum.clone()
        if world_size > 1:
            dist.all_reduce(data_sum, op=dist.ReduceOp.SUM)
        return (data_sum / total_observations).item()

    history = [data_loss()]
    for iteration in range(args.iterations):
        if args.optimizer == "lbfgs":
            optimizer.step(closure)
        else:
            closure()
            optimizer.step()
        history.append(data_loss())
        if rank == 0 and (iteration % max(1, args.iterations // 10) == 0 or iteration == args.iterations - 1):
            print(f"iteration {iteration + 1:03d}/{args.iterations} data={history[-1]:.6e}")
    return history


def horizontal_error_km(catalog, truth):
    dlon = (catalog.longitude.to_numpy() - truth.longitude.to_numpy()) * np.cos(np.deg2rad(truth.latitude.to_numpy()))
    dlat = catalog.latitude.to_numpy() - truth.latitude.to_numpy()
    return KM_PER_DEGREE * np.hypot(dlon, dlat)


def inverted_catalog(events_initial, tomography):
    catalog = events_initial.copy()
    event_loc = tomography.event_loc.detach().numpy()
    correction = tomography.event_time_correction.detach().numpy()
    catalog["longitude"], catalog["latitude"], catalog["depth_km"] = event_loc[:, 0], event_loc[:, 1], event_loc[:, 2]
    catalog["event_time"] = [
        (pd.Timestamp(time) + pd.Timedelta(seconds=float(shift))).isoformat(timespec="milliseconds")
        for time, shift in zip(events_initial.event_time, correction)
    ]
    catalog["dt0_s"] = correction
    return catalog


def print_summary(true, initial, model, events, events_initial, inverted):
    for phase in ("vp", "vs"):
        recovered = getattr(model, phase).detach()
        print(
            f"{phase.upper()} mean |error|: initial={(initial[phase] - true[phase]).abs().mean():.4f} "
            f"recovered={(recovered - true[phase]).abs().mean():.4f} km/s"
        )
    seconds = lambda catalog: (pd.to_datetime(catalog.event_time) - pd.to_datetime(events.event_time)).dt.total_seconds().abs().mean()
    print(f"event horizontal error (km): initial={horizontal_error_km(events_initial, events).mean():.3f} recovered={horizontal_error_km(inverted, events).mean():.3f}")
    print(f"event depth error (km): initial={(events_initial.depth_km - events.depth_km).abs().mean():.3f} recovered={(inverted.depth_km - events.depth_km).abs().mean():.3f}")
    print(f"origin-time error (s): initial={seconds(events_initial):.3f} recovered={seconds(inverted):.3f}")


def plot_velocity(true, initial, model, path):
    depth_index = int(torch.argmin((model.depth - 15.0).abs()))
    extent = [model.lon[0], model.lon[-1], model.lat[0], model.lat[-1]]
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for row, phase in enumerate(("vp", "vs")):
        truth = (true[phase][depth_index] - initial[phase][depth_index]) / initial[phase][depth_index]
        recovered = (getattr(model, phase).detach()[depth_index] - initial[phase][depth_index]) / initial[phase][depth_index]
        limit = max(truth.abs().max().item(), recovered.abs().max().item(), 1e-6)
        for axis, title, field in zip(axes[row], ("true checkerboard", "recovered perturbation"), (truth, recovered)):
            image = axis.imshow(field, origin="lower", extent=extent, cmap="seismic", vmin=-limit, vmax=limit)
            axis.set(title=f"{phase.upper()} {title}", xlabel="longitude (deg)", ylabel="latitude (deg)")
            figure.colorbar(image, ax=axis, label="relative velocity")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_events(events, events_initial, inverted, history, path):
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    axes[0].semilogy(history, "o-", color="tab:blue")
    axes[0].set(title="Data misfit", xlabel="iteration", ylabel="phase-time MSE (s²)")
    axes[0].grid(alpha=0.3)
    axes[1].scatter(events.longitude, events.latitude, marker="*", s=60, color="k", label="true")
    axes[1].scatter(events_initial.longitude, events_initial.latitude, marker="x", color="tab:gray", label="initial")
    axes[1].scatter(inverted.longitude, inverted.latitude, marker="o", facecolors="none", edgecolors="tab:red", label="inverted")
    axes[1].set(title="Epicenters", xlabel="longitude (deg)", ylabel="latitude (deg)")
    axes[1].legend()
    axes[2].plot(events_initial.depth_km - events.depth_km, "x", color="tab:gray", label="depth error initial (km)")
    axes[2].plot(inverted.depth_km - events.depth_km, "o", color="tab:red", label="depth error inverted (km)")
    axes[2].plot(inverted.dt0_s, ".", color="tab:green", label="origin-time correction (s)")
    axes[2].axhline(0.0, color="k", lw=0.8)
    axes[2].set(title="Depth and origin-time corrections", xlabel="event index")
    axes[2].legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    rank, world_size = distributed_context()
    try:
        initial, true, stations, events, events_initial, picks = load_data(args.data_dir)
        model = VelocityModel(**initial, trainable=True)
        tomography = Tomography(
            model, args.lambda_vp, args.lambda_vs, args.alpha_vp, args.alpha_vs,
            event_loc=events_initial[["longitude", "latitude", "depth_km"]].to_numpy(),
        )
        parameters = configure_trainable(tomography, args.trainable)
        groups = prepare_station_groups(stations, events_initial, picks, model, args.spacing, rank, world_size)
        if not groups:
            raise ValueError("NPROC cannot exceed the number of stations with picks")
        if rank == 0:
            print(f"{len(stations)} stations, {len(events)} events, {len(picks)} picks; world_size={world_size}; optimizer={args.optimizer}")
            print("Optimizing parameters:\n" + "\n".join(f"  {name}: {tuple(parameter.shape)}" for name, parameter in tomography.named_parameters() if parameter.requires_grad))

        history = optimize(tomography, groups, parameters, args, len(picks), rank, world_size)

        if rank == 0:
            args.results_dir.mkdir(exist_ok=True)
            args.figures_dir.mkdir(exist_ok=True)
            inverted = inverted_catalog(events_initial, tomography)
            torch.save(
                {"lon": model.lon, "lat": model.lat, "depth": model.depth, "vp": model.vp.detach(), "vs": model.vs.detach(), "loss_history": history},
                args.results_dir / "model_inverted.pt",
            )
            inverted.to_csv(args.results_dir / "events_inverted.csv", index=False)
            plot_velocity(true, initial, model, args.figures_dir / "inversion.png")
            plot_events(events, events_initial, inverted, history, args.figures_dir / "events.png")
            print_summary(true, initial, model, events, events_initial, inverted)
            print(f"saved results to {args.results_dir} and figures to {args.figures_dir}; misfit {history[0]:.3e} -> {history[-1]:.3e}")
        if world_size > 1:
            dist.barrier()
    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
