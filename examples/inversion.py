"""Invert the pre-generated synthetic picks, serially or under ``torchrun``."""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from adtomo import ForwardGrid, Tomography, VelocityModel


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"


def prepare_pick_groups(stations, events, picks, model, spacing):
    events_by_id = events.set_index("event_id")
    stations_by_id = stations.set_index("station_id")
    groups = []
    for station_id, station_picks in picks.groupby("station_id", sort=False):
        station = stations_by_id.loc[station_id]
        event_ids = pd.unique(station_picks.event_id)
        station_events = events_by_id.loc[event_ids]
        station_spherical = torch.tensor([station.longitude, station.latitude, station.depth_km], dtype=torch.float64)
        events_spherical = torch.tensor(station_events[["longitude", "latitude", "depth_km"]].to_numpy(), dtype=torch.float64)
        grid = ForwardGrid(station_spherical, events_spherical, model, spacing=spacing)
        phase_groups = []
        for phase, phase_picks in station_picks.groupby("phase_type", sort=False):
            event_indices = torch.tensor(pd.Index(event_ids).get_indexer(phase_picks.event_id), dtype=torch.long)
            origin_times = pd.to_datetime(phase_picks.event_id.map(events_by_id.event_time))
            observed = torch.tensor((pd.to_datetime(phase_picks.phase_time) - origin_times).dt.total_seconds().to_numpy(), dtype=torch.float64)
            phase_groups.append((phase, event_indices, observed))
        groups.append((grid, phase_groups))
    return groups


def plot_result(true, initial, model, path):
    depth_index = int(torch.argmin((model.depth - 15.0).abs()))
    extent = [model.lon[0], model.lon[-1], model.lat[0], model.lat[-1]]
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for row, phase in enumerate(("vp", "vs")):
        truth = (true[phase][depth_index] - initial[phase][depth_index]) / initial[phase][depth_index]
        recovered = (getattr(model, phase).detach()[depth_index] - initial[phase][depth_index]) / initial[phase][depth_index]
        limit = max(truth.abs().max().item(), recovered.abs().max().item())
        for axis, title, field in zip(axes[row], ("true checkerboard", "recovered perturbation"), (truth, recovered)):
            image = axis.imshow(field, origin="lower", extent=extent, cmap="seismic", vmin=-limit, vmax=limit)
            axis.set(title=f"{phase.upper()} {title}", xlabel="longitude (deg)", ylabel="latitude (deg)")
            figure.colorbar(image, ax=axis, label="relative velocity")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--spacing", type=float, default=2.0)
    parser.add_argument("--lambda-vp", type=float, default=0.0)
    parser.add_argument("--lambda-vs", type=float, default=0.0)
    parser.add_argument("--alpha-vp", type=float, default=0.0)
    parser.add_argument("--alpha-vs", type=float, default=0.0)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    if args.iterations < 1 or args.learning_rate <= 0 or args.spacing <= 0:
        raise ValueError("iterations, learning rate, and spacing must be positive")

    data_dir = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
    results_dir = args.results_dir if args.results_dir.is_absolute() else ROOT / args.results_dir
    figures_dir = args.figures_dir if args.figures_dir.is_absolute() else ROOT / args.figures_dir
    required = [data_dir / name for name in ("model_initial.pt", "model_true.pt", "stations.csv", "events.csv", "picks.csv")]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("run 00_gen_velocity.py through 03_gen_picks.py first")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("gloo")
        rank = dist.get_rank()
    else:
        rank = 0

    try:
        initial = torch.load(data_dir / "model_initial.pt", weights_only=True)
        true = torch.load(data_dir / "model_true.pt", weights_only=True)
        stations = pd.read_csv(data_dir / "stations.csv", dtype={"station_id": str})
        events = pd.read_csv(data_dir / "events.csv", dtype={"event_id": str})
        picks = pd.read_csv(data_dir / "picks.csv", dtype={"event_id": str, "station_id": str})
        grids = VelocityModel(**initial, trainable=False)
        groups = prepare_pick_groups(stations, events, picks, grids, args.spacing)
        local_groups = groups[rank::world_size]
        if not local_groups:
            raise ValueError("NPROC cannot exceed the number of station groups")

        model = VelocityModel(**initial, trainable=True)
        tomography = Tomography(model, args.lambda_vp, args.lambda_vs, args.alpha_vp, args.alpha_vs)
        ddp = DistributedDataParallel(tomography) if distributed else None
        optimizer = torch.optim.Adam(tomography.parameters(), lr=args.learning_rate)
        total_observations = len(picks)

        for iteration in range(args.iterations):
            optimizer.zero_grad()
            if distributed:
                loss = ddp(local_groups, data_scale=world_size / total_observations)
            else:
                loss = tomography(groups)
            loss.backward()
            optimizer.step()
            if distributed:
                data_sum = tomography.data_sum.clone()
                dist.all_reduce(data_sum)
                data_loss = data_sum / total_observations
                total_loss = data_loss + tomography.regularization_loss
            else:
                data_loss, total_loss = tomography.data_loss, tomography.total_loss
            if rank == 0:
                print(f"iteration {iteration + 1:03d}/{args.iterations} data={data_loss.item():.6f} total={total_loss.item():.6f}")

        if distributed:
            dist.barrier()
        if rank == 0:
            results_dir.mkdir(exist_ok=True)
            figures_dir.mkdir(exist_ok=True)
            torch.save({"lon": model.lon, "lat": model.lat, "depth": model.depth, "vp": model.vp.detach(), "vs": model.vs.detach()}, results_dir / "model_inverted.pt")
            plot_result(true, initial, model, figures_dir / "inversion.png")
            print(f"saved model to {results_dir / 'model_inverted.pt'}")
        if distributed:
            dist.barrier()
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
