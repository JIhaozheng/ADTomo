"""Invert the pre-generated synthetic picks, serially or under ``torchrun``."""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from adtomo import ForwardGrid, Tomography, VelocityModel, predict_travel_times, smoothness


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
REQUIRED_INPUTS = (
    "model_initial.pt",
    "model_true.pt",
    "stations.csv",
    "events.csv",
    "picks.csv",
)


def distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return False, 0, 1
    dist.init_process_group("gloo")
    return True, dist.get_rank(), dist.get_world_size()


def resolve_example_path(path):
    """Resolve relative CLI paths from the examples directory, not the shell cwd."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_data(data_dir):
    missing = [data_dir / name for name in REQUIRED_INPUTS if not (data_dir / name).is_file()]
    if missing:
        paths = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"missing inversion inputs:\n  {paths}\n"
            "generate them first with:\n"
            "  python examples/00_gen_velocity.py\n"
            "  python examples/01_gen_stations.py\n"
            "  python examples/02_gen_events.py\n"
            "  python examples/03_gen_picks.py"
        )
    return (
        torch.load(data_dir / "model_initial.pt", weights_only=True),
        torch.load(data_dir / "model_true.pt", weights_only=True),
        pd.read_csv(data_dir / "stations.csv", dtype={"station_id": str}),
        pd.read_csv(data_dir / "events.csv", dtype={"event_id": str}),
        pd.read_csv(data_dir / "picks.csv", dtype={"event_id": str, "station_id": str}),
    )


def prepare_pick_groups(stations, events, picks, model, spacing):
    """Build one station-aligned ForwardGrid and its observations per station."""
    events_by_id = events.set_index("event_id")
    stations_by_id = stations.set_index("station_id")
    if not events_by_id.index.is_unique or not stations_by_id.index.is_unique:
        raise ValueError("event_id and station_id values must be unique")

    groups = []
    for station_id, station_picks in picks.groupby("station_id", sort=False):
        if station_id not in stations_by_id.index:
            raise ValueError(f"pick references unknown station_id {station_id!r}")
        event_ids = pd.unique(station_picks.event_id)
        if not pd.Index(event_ids).isin(events_by_id.index).all():
            raise ValueError(f"picks for {station_id!r} reference an unknown event_id")

        station = stations_by_id.loc[station_id]
        station_spherical = torch.tensor(
            [station.longitude, station.latitude, station.depth_km], dtype=torch.float64
        )
        station_events = events_by_id.loc[event_ids]
        events_spherical = torch.tensor(
            station_events[["longitude", "latitude", "depth_km"]].to_numpy(),
            dtype=torch.float64,
        )
        grid = ForwardGrid(station_spherical, events_spherical, model, spacing=spacing)

        phase_groups = []
        for phase, phase_picks in station_picks.groupby("phase_type", sort=False):
            event_indices = torch.tensor(
                pd.Index(event_ids).get_indexer(phase_picks.event_id), dtype=torch.long
            )
            event_times = pd.to_datetime(phase_picks.event_id.map(events_by_id.event_time))
            observed = torch.tensor(
                (pd.to_datetime(phase_picks.phase_time) - event_times).dt.total_seconds().to_numpy(),
                dtype=torch.float64,
            )
            phase_groups.append((phase, event_indices, observed))
        groups.append((grid, phase_groups))
    if not groups:
        raise ValueError("picks.csv contains no station groups")
    return groups


def squared_residual_sum(model, groups):
    residuals = [
        predict_travel_times(model, grid, phase, event_indices) - observed
        for grid, phase_groups in groups
        for phase, event_indices, observed in phase_groups
    ]
    return torch.cat(residuals).square().sum()


def observation_count(groups):
    return sum(len(observed) for _, phase_groups in groups for _, _, observed in phase_groups)


def regularization_loss(tomography):
    model = tomography.model
    dvp = model.vp - tomography.vp0
    dvs = model.vs - tomography.vs0
    return (
        tomography.lambda_vp * smoothness(dvp, model.lon, model.lat, model.depth)
        + tomography.lambda_vs * smoothness(dvs, model.lon, model.lat, model.depth)
        + tomography.alpha_vp * dvp.square().mean()
        + tomography.alpha_vs * dvs.square().mean()
    )


class DistributedObjective(nn.Module):
    """DDP objective whose averaged gradients equal the serial global objective."""

    def __init__(self, tomography, world_size, total_observations):
        super().__init__()
        self.tomography = tomography
        self.world_size = world_size
        self.total_observations = total_observations

    def forward(self, local_groups):
        data_loss = (
            self.world_size
            * squared_residual_sum(self.tomography.model, local_groups)
            / self.total_observations
        )
        return data_loss + regularization_loss(self.tomography)


def global_metrics(objective, local_groups):
    """Return serial-equivalent global data and total losses on every rank."""
    residual_sum = squared_residual_sum(objective.tomography.model, local_groups).detach()
    dist.all_reduce(residual_sum, op=dist.ReduceOp.SUM)
    data_loss = residual_sum / objective.total_observations
    total_loss = data_loss + regularization_loss(objective.tomography).detach()
    return data_loss, total_loss


def relative_error(actual, expected):
    denominator = expected.norm().clamp_min(torch.finfo(actual.dtype).tiny)
    return (actual - expected).norm() / denominator


def make_tomography(initial, args):
    model = VelocityModel(**initial, trainable=True)
    tomography = Tomography(
        model,
        lambda_vp=args.lambda_vp,
        lambda_vs=args.lambda_vs,
        alpha_vp=args.alpha_vp,
        alpha_vs=args.alpha_vs,
    )
    return model, tomography


def validate_ddp(initial, groups, local_groups, objective, ddp, args, rank):
    """Compare the distributed objective, gradients, and one update with serial."""
    reference = make_tomography(initial, args)[1] if rank == 0 else None
    optimizer = torch.optim.SGD(ddp.parameters(), lr=args.learning_rate)
    optimizer.zero_grad()
    ddp(local_groups).backward()
    data_loss, total_loss = global_metrics(objective, local_groups)

    if reference is not None:
        reference_loss = reference(groups)
        reference_loss.backward()
        if not torch.allclose(data_loss, reference.data_loss, atol=1e-12):
            raise AssertionError("DDP and serial data losses differ")
        if not torch.allclose(total_loss, reference.total_loss, atol=1e-12):
            raise AssertionError("DDP and serial total losses differ")
        vp_gradient_error = relative_error(
            objective.tomography.model.vp.grad, reference.model.vp.grad
        ).item()
        vs_gradient_error = relative_error(
            objective.tomography.model.vs.grad, reference.model.vs.grad
        ).item()
        if vp_gradient_error >= 1e-11 or vs_gradient_error >= 1e-11:
            raise AssertionError(
                f"DDP gradient mismatch: Vp={vp_gradient_error:.3e}, Vs={vs_gradient_error:.3e}"
            )

    optimizer.step()
    data_loss, total_loss = global_metrics(objective, local_groups)
    if reference is not None:
        torch.optim.SGD(reference.parameters(), lr=args.learning_rate).step()
        vp_update_error = relative_error(
            objective.tomography.model.vp.detach(), reference.model.vp.detach()
        ).item()
        vs_update_error = relative_error(
            objective.tomography.model.vs.detach(), reference.model.vs.detach()
        ).item()
        if vp_update_error >= 1e-11 or vs_update_error >= 1e-11:
            raise AssertionError(
                f"DDP update mismatch: Vp={vp_update_error:.3e}, Vs={vs_update_error:.3e}"
            )
        print(
            "DDP validation passed: serial objective, gradients, and one SGD update agree; "
            f"gradient errors={vp_gradient_error:.3e}/{vs_gradient_error:.3e}, "
            f"update errors={vp_update_error:.3e}/{vs_update_error:.3e}"
        )
    return [data_loss.item()], [total_loss.item()]


def run_inversion(initial, groups, local_groups, args, distributed, rank, world_size):
    model, tomography = make_tomography(initial, args)
    if not distributed:
        optimizer = torch.optim.Adam(tomography.parameters(), lr=args.learning_rate)
        data_history = []
        total_history = []
        for iteration in range(args.iterations):
            optimizer.zero_grad()
            loss = tomography(groups)
            data_history.append(tomography.data_loss.item())
            total_history.append(loss.item())
            loss.backward()
            optimizer.step()
            print(
                f"iteration {iteration + 1:03d}/{args.iterations} "
                f"data={data_history[-1]:.6f} total={total_history[-1]:.6f}"
            )
        with torch.no_grad():
            final_loss = tomography(groups)
        data_history.append(tomography.data_loss.item())
        total_history.append(final_loss.item())
        return model, data_history, total_history

    if not local_groups:
        raise ValueError("NPROC cannot exceed the number of station groups")
    objective = DistributedObjective(tomography, world_size, observation_count(groups))
    ddp = DistributedDataParallel(objective)
    if args.validate_ddp:
        histories = validate_ddp(initial, groups, local_groups, objective, ddp, args, rank)
        return model, *histories

    optimizer = torch.optim.Adam(ddp.parameters(), lr=args.learning_rate)
    data_history = []
    total_history = []
    for iteration in range(args.iterations):
        optimizer.zero_grad()
        ddp(local_groups).backward()
        optimizer.step()
        data_loss, total_loss = global_metrics(objective, local_groups)
        data_history.append(data_loss.item())
        total_history.append(total_loss.item())
        if rank == 0:
            print(
                f"iteration {iteration + 1:03d}/{args.iterations} "
                f"data={data_history[-1]:.6f} total={total_history[-1]:.6f}"
            )
    return model, data_history, total_history


def plot_inversion(model_true, model_initial, model, path):
    """Compare true and recovered relative perturbations at one depth."""
    depth_index = int(torch.argmin((model.depth - 15.0).abs()))
    depth_km = model.depth[depth_index].item()
    extent = [model.lon[0].item(), model.lon[-1].item(), model.lat[0].item(), model.lat[-1].item()]
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for row, name in enumerate(("vp", "vs")):
        true_relative = (
            model_true[name][depth_index] - model_initial[name][depth_index]
        ) / model_initial[name][depth_index]
        recovered_relative = (
            getattr(model, name).detach()[depth_index] - model_initial[name][depth_index]
        ) / model_initial[name][depth_index]
        limit = max(true_relative.abs().max().item(), recovered_relative.abs().max().item())
        for axis, title, field in zip(
            axes[row], ("true checkerboard", "recovered perturbation"), (true_relative, recovered_relative)
        ):
            image = axis.imshow(
                field,
                origin="lower",
                extent=extent,
                cmap="seismic",
                vmin=-limit,
                vmax=limit,
            )
            axis.set(
                title=f"{name.upper()} {title}",
                xlabel="longitude (deg)",
                ylabel="latitude (deg)",
            )
            figure.colorbar(image, ax=axis, label="relative velocity")
    figure.suptitle(f"Checkerboard recovery at {depth_km:.1f} km depth")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--spacing", type=float, default=5.0, help="forward-grid spacing in km")
    parser.add_argument("--lambda-vp", type=float, default=0.0, help="Vp smoothness weight")
    parser.add_argument("--lambda-vs", type=float, default=0.0, help="Vs smoothness weight")
    parser.add_argument("--alpha-vp", type=float, default=0.0, help="Vp damping weight")
    parser.add_argument("--alpha-vs", type=float, default=0.0, help="Vs damping weight")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES)
    parser.add_argument(
        "--validate-ddp",
        action="store_true",
        help="under torchrun, compare one distributed SGD step with serial",
    )
    return parser.parse_args()


def validate_arguments(args):
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if args.learning_rate <= 0 or args.spacing <= 0:
        raise ValueError("learning rate and spacing must be positive")
    weights = (args.lambda_vp, args.lambda_vs, args.alpha_vp, args.alpha_vs)
    if any(weight < 0 for weight in weights):
        raise ValueError("regularization weights must be non-negative")


def main():
    args = parse_arguments()
    validate_arguments(args)
    args.data_dir = resolve_example_path(args.data_dir)
    args.results_dir = resolve_example_path(args.results_dir)
    args.figures_dir = resolve_example_path(args.figures_dir)
    distributed, rank, world_size = distributed_context()
    try:
        if args.validate_ddp and not distributed:
            raise ValueError("--validate-ddp requires NPROC greater than 1")
        initial, true, stations, events, picks = load_data(args.data_dir)
        grid_model = VelocityModel(**initial, trainable=False)
        groups = prepare_pick_groups(stations, events, picks, grid_model, args.spacing)
        local_groups = groups[rank::world_size]
        if rank == 0:
            distribution = [len(groups[index::world_size]) for index in range(world_size)]
            print(f"world_size={world_size}; station groups per rank={distribution}")
        model, data_history, total_history = run_inversion(
            initial, groups, local_groups, args, distributed, rank, world_size
        )
        if distributed:
            dist.barrier()
        if rank == 0:
            args.results_dir.mkdir(parents=True, exist_ok=True)
            args.figures_dir.mkdir(parents=True, exist_ok=True)
            model_path = args.results_dir / "model_inverted.pt"
            figure_path = args.figures_dir / "inversion.png"
            torch.save(
                {
                    "lon": model.lon,
                    "lat": model.lat,
                    "depth": model.depth,
                    "vp": model.vp.detach(),
                    "vs": model.vs.detach(),
                },
                model_path,
            )
            plot_inversion(true, initial, model, figure_path)
            print(
                f"saved inverted model to {model_path} and diagnostic to {figure_path}; "
                f"final data/total loss={data_history[-1]:.6f}/{total_history[-1]:.6f}"
            )
        if distributed:
            dist.barrier()
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
