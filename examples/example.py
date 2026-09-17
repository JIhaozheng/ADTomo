"""Station-group checkerboard inversion, serial or under ``torchrun``."""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from adtomo import ForwardGrid, Tomography, VelocityModel, predict_travel_times, smoothness


ROOT = Path(__file__).resolve().parent
DEFAULT_BACKGROUND_MODEL = ROOT.parent / "example_real" / "data" / "model_initial.pt"


def distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return False, 0, 1
    dist.init_process_group("gloo")
    return True, dist.get_rank(), dist.get_world_size()


def make_checkerboard(shape, block_lon, block_lat, block_depth, dtype):
    nz, ny, nx = shape
    checker = (
        torch.arange(nx)[None, None, :] // block_lon
        + torch.arange(ny)[None, :, None] // block_lat
        + torch.arange(nz)[:, None, None] // block_depth
    ) % 2
    return (2 * checker - 1).to(dtype=dtype)


def generate_models(background_path, data_dir, amplitude, block_lon, block_lat, block_depth):
    if not background_path.is_file():
        raise FileNotFoundError(
            f"background model not found: {background_path}; pass --background-model with a .pt model file"
        )
    background = torch.load(background_path, weights_only=True)
    required = {"lon", "lat", "depth", "vp", "vs"}
    missing = required.difference(background)
    if missing:
        raise ValueError(f"background model is missing {sorted(missing)}")
    initial = {name: background[name].detach().clone() for name in required}
    checker = make_checkerboard(
        initial["vp"].shape, block_lon, block_lat, block_depth, initial["vp"].dtype
    )
    true = {name: value.detach().clone() for name, value in initial.items()}
    true["vp"] = initial["vp"] * (1.0 + amplitude * checker)
    true["vs"] = initial["vs"] * (1.0 + amplitude * checker)
    if true["vp"].min() <= 0 or true["vs"].min() <= 0:
        raise ValueError("checkerboard amplitude makes Vp or Vs non-positive")
    relative_vp = (true["vp"] - initial["vp"]) / initial["vp"]
    relative_vs = (true["vs"] - initial["vs"]) / initial["vs"]
    assert torch.allclose(relative_vp.abs(), torch.full_like(relative_vp, amplitude))
    assert torch.allclose(relative_vs.abs(), torch.full_like(relative_vs, amplitude))
    assert relative_vp.min() < 0 < relative_vp.max()
    torch.save(initial, data_dir / "model_initial.pt")
    torch.save(true, data_dir / "model_true.pt")
    return initial, true


def generate_stations(data_dir):
    stations = pd.DataFrame(
        [
            ("STA01", -120.28, 34.78, 0.0),
            ("STA02", -119.72, 34.82, 0.0),
            ("STA03", -120.22, 35.27, 0.0),
            ("STA04", -119.76, 35.23, 0.0),
        ],
        columns=["station_id", "longitude", "latitude", "depth_km"],
    )
    stations.to_csv(data_dir / "stations.csv", index=False)
    return stations


def generate_events(data_dir, count, seed, background):
    rng = np.random.default_rng(seed)
    origin = pd.Timestamp("2026-09-13T12:00:00.000")
    lon_min, lon_max = max(-120.7, background["lon"][0].item() + 0.2), min(-119.2, background["lon"][-1].item() - 0.2)
    lat_min, lat_max = max(34.3, background["lat"][0].item() + 0.2), min(35.8, background["lat"][-1].item() - 0.2)
    depth_min, depth_max = max(2.0, background["depth"][0].item() + 2.0), min(22.0, background["depth"][-1].item() - 2.0)
    if lon_min >= lon_max or lat_min >= lat_max or depth_min >= depth_max:
        raise ValueError("background model does not cover the example catalog region with its required halo")
    events = pd.DataFrame(
        {
            "event_id": [f"EV{i:03d}" for i in range(count)],
            "event_time": [(origin + pd.Timedelta(seconds=30 * i)).isoformat(timespec="milliseconds") for i in range(count)],
            "longitude": rng.uniform(lon_min, lon_max, count),
            "latitude": rng.uniform(lat_min, lat_max, count),
            "depth_km": rng.uniform(depth_min, depth_max, count),
        }
    )
    events.to_csv(data_dir / "events.csv", index=False)
    return events


def generate_picks(data_dir, stations, events, true, spacing):
    model = VelocityModel(**true, trainable=False)
    events_spherical = torch.tensor(events[["longitude", "latitude", "depth_km"]].values, dtype=torch.float64)
    picks = []
    with torch.no_grad():
        for station in stations.itertuples(index=False):
            station_spherical = torch.tensor(
                [station.longitude, station.latitude, station.depth_km], dtype=torch.float64
            )
            grid = ForwardGrid(station_spherical, events_spherical, model, spacing=spacing)
            for phase in ("P", "S"):
                travel_times = predict_travel_times(model, grid, phase)
                for event, travel_time in zip(events.itertuples(index=False), travel_times.tolist()):
                    picks.append(
                        {
                            "event_id": event.event_id,
                            "station_id": station.station_id,
                            "phase_type": phase,
                            "phase_time": (pd.Timestamp(event.event_time) + pd.Timedelta(seconds=travel_time)).isoformat(
                                timespec="milliseconds"
                            ),
                        }
                    )
    picks = pd.DataFrame(picks)
    picks.to_csv(data_dir / "picks.csv", index=False)
    return picks


def load_data(data_dir):
    required = ["model_initial.pt", "model_true.pt", "stations.csv", "events.csv", "picks.csv"]
    missing = [str(data_dir / name) for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("--skip-prepare requires: " + ", ".join(missing))
    return (
        torch.load(data_dir / "model_initial.pt", weights_only=True),
        torch.load(data_dir / "model_true.pt", weights_only=True),
        pd.read_csv(data_dir / "stations.csv", dtype={"station_id": str}),
        pd.read_csv(data_dir / "events.csv", dtype={"event_id": str}),
        pd.read_csv(data_dir / "picks.csv", dtype={"event_id": str, "station_id": str}),
    )


def prepare_pick_groups(stations, events, picks, model, spacing):
    events_by_id = events.set_index("event_id")
    if not events_by_id.index.is_unique:
        raise ValueError("event_id values must be unique")
    stations_by_id = stations.set_index("station_id")
    groups = []
    for station_id, station_picks in picks.groupby("station_id", sort=False):
        if station_id not in stations_by_id.index:
            raise ValueError(f"pick references unknown station_id {station_id!r}")
        event_ids = pd.unique(station_picks.event_id)
        if not pd.Index(event_ids).isin(events_by_id.index).all():
            raise ValueError(f"picks for {station_id!r} reference an unknown event_id")
        station = stations_by_id.loc[station_id]
        station_events = events_by_id.loc[event_ids]
        station_spherical = torch.tensor([station.longitude, station.latitude, station.depth_km], dtype=torch.float64)
        events_spherical = torch.tensor(
            station_events[["longitude", "latitude", "depth_km"]].values, dtype=torch.float64
        )
        grid = ForwardGrid(station_spherical, events_spherical, model, spacing=spacing)
        phase_groups = []
        for phase, phase_picks in station_picks.groupby("phase_type", sort=False):
            event_indices = torch.tensor(pd.Index(event_ids).get_indexer(phase_picks.event_id), dtype=torch.long)
            event_times = pd.to_datetime(phase_picks.event_id.map(events_by_id.event_time))
            phase_dt = torch.tensor(
                (pd.to_datetime(phase_picks.phase_time) - event_times).dt.total_seconds().to_numpy(),
                dtype=torch.float64,
            )
            phase_groups.append((phase, event_indices, phase_dt))
        groups.append((grid, phase_groups))
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


class DistributedObjective(nn.Module):
    """DDP loss scaled so averaged gradients equal the serial global MSE."""

    def __init__(self, tomography, world_size, total_observations):
        super().__init__()
        self.tomography = tomography
        self.world_size = world_size
        self.total_observations = total_observations

    def forward(self, local_groups):
        model = self.tomography.model
        data_loss = self.world_size * squared_residual_sum(model, local_groups) / self.total_observations
        dvp, dvs = model.vp - self.tomography.vp0, model.vs - self.tomography.vs0
        regularization = (
            self.tomography.lambda_vp * smoothness(dvp, model.lon, model.lat, model.depth)
            + self.tomography.lambda_vs * smoothness(dvs, model.lon, model.lat, model.depth)
            + self.tomography.alpha_vp * dvp.square().mean()
            + self.tomography.alpha_vs * dvs.square().mean()
        )
        return data_loss + regularization


def global_metrics(objective, local_groups):
    local_sum = squared_residual_sum(objective.tomography.model, local_groups).detach()
    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
    model = objective.tomography.model
    dvp, dvs = model.vp - objective.tomography.vp0, model.vs - objective.tomography.vs0
    regularization = (
        objective.tomography.lambda_vp * smoothness(dvp, model.lon, model.lat, model.depth)
        + objective.tomography.lambda_vs * smoothness(dvs, model.lon, model.lat, model.depth)
        + objective.tomography.alpha_vp * dvp.square().mean()
        + objective.tomography.alpha_vs * dvs.square().mean()
    )
    data_loss = local_sum / objective.total_observations
    return data_loss, data_loss + regularization


def relative_error(actual, expected):
    return (actual - expected).norm() / expected.norm().clamp_min(torch.finfo(actual.dtype).tiny)


def validate_ddp(model, initial, groups, local_groups, objective, distributed, learning_rate, rank):
    if not distributed:
        raise ValueError("--validate-ddp requires NPROC greater than 1")
    reference = Tomography(VelocityModel(**initial, trainable=True)) if rank == 0 else None
    optimizer = torch.optim.SGD(distributed.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    loss = distributed(local_groups)
    loss.backward()
    data_loss, total_loss = global_metrics(objective, local_groups)
    if rank == 0:
        reference_loss = reference(groups)
        reference_loss.backward()
        assert torch.allclose(data_loss, reference.data_loss, atol=1e-12)
        assert torch.allclose(total_loss, reference.total_loss, atol=1e-12)
        vp_gradient_error = relative_error(model.vp.grad, reference.model.vp.grad).item()
        vs_gradient_error = relative_error(model.vs.grad, reference.model.vs.grad).item()
        assert vp_gradient_error < 1e-11 and vs_gradient_error < 1e-11
    optimizer.step()
    data_loss, total_loss = global_metrics(objective, local_groups)
    if rank == 0:
        torch.optim.SGD(reference.parameters(), lr=learning_rate).step()
        vp_update_error = relative_error(model.vp.detach(), reference.model.vp.detach()).item()
        vs_update_error = relative_error(model.vs.detach(), reference.model.vs.detach()).item()
        assert vp_update_error < 1e-11 and vs_update_error < 1e-11
        print(
            "DDP validation passed: "
            f"Vp/Vs gradient errors={vp_gradient_error:.3e}/{vs_gradient_error:.3e}; "
            f"update errors={vp_update_error:.3e}/{vs_update_error:.3e}"
        )
    return model, [data_loss.item()], [total_loss.item()]


def run_inversion(initial, groups, local_groups, args, distributed, rank, world_size):
    model = VelocityModel(**initial, trainable=True)
    tomography = Tomography(model)
    if not distributed:
        optimizer = torch.optim.Adam(tomography.parameters(), lr=args.learning_rate)
        data_history, total_history = [], []
        for iteration in range(args.iterations):
            optimizer.zero_grad()
            loss = tomography(groups)
            data_history.append(tomography.data_loss.item())
            total_history.append(loss.item())
            loss.backward()
            optimizer.step()
            print(f"iteration {iteration + 1:02d}/{args.iterations} data={data_history[-1]:.6f} total={total_history[-1]:.6f}")
        with torch.no_grad():
            final_loss = tomography(groups)
        return model, data_history + [tomography.data_loss.item()], total_history + [final_loss.item()]

    if not local_groups:
        raise ValueError("NPROC cannot exceed the number of station groups")
    objective = DistributedObjective(tomography, world_size, observation_count(groups))
    ddp = DistributedDataParallel(objective)
    if args.validate_ddp:
        return validate_ddp(model, initial, groups, local_groups, objective, ddp, args.learning_rate, rank)
    optimizer = torch.optim.Adam(ddp.parameters(), lr=args.learning_rate)
    data_history, total_history = [], []
    for iteration in range(args.iterations):
        optimizer.zero_grad()
        loss = ddp(local_groups)
        loss.backward()
        optimizer.step()
        data_loss, total_loss = global_metrics(objective, local_groups)
        data_history.append(data_loss.item())
        total_history.append(total_loss.item())
        if rank == 0:
            print(f"iteration {iteration + 1:02d}/{args.iterations} data={data_loss.item():.6f} total={total_loss.item():.6f}")
    return model, data_history, total_history


def plot_results(true, initial, model, data_history, total_history, path):
    depth_index = int(torch.argmin((model.depth - 15.0).abs()))
    extent = [model.lon[0].item(), model.lon[-1].item(), model.lat[0].item(), model.lat[-1].item()]
    figure, axes = plt.subplots(2, 6, figsize=(20, 7), constrained_layout=True)
    for row, name in enumerate(("vp", "vs")):
        fields = [true[name][depth_index], initial[name][depth_index], getattr(model, name).detach()[depth_index]]
        vmin, vmax = min(field.min().item() for field in fields), max(field.max().item() for field in fields)
        for column, (title, field) in enumerate(zip(("true", "initial", "inverted"), fields)):
            image = axes[row, column].imshow(field, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
            axes[row, column].set(title=f"{name.upper()} {title}", xlabel="longitude (deg)", ylabel="latitude (deg)")
            figure.colorbar(image, ax=axes[row, column], shrink=0.8, label="km/s")
        true_relative = (true[name][depth_index] - initial[name][depth_index]) / initial[name][depth_index]
        recovered_relative = (getattr(model, name).detach()[depth_index] - initial[name][depth_index]) / initial[name][depth_index]
        limit = max(true_relative.abs().max().item(), recovered_relative.abs().max().item())
        for column, (title, field) in enumerate((("true perturbation", true_relative), ("recovered perturbation", recovered_relative)), 3):
            image = axes[row, column].imshow(field, origin="lower", extent=extent, cmap="seismic", vmin=-limit, vmax=limit)
            axes[row, column].set(title=f"{name.upper()} {title}", xlabel="longitude (deg)", ylabel="latitude (deg)")
            figure.colorbar(image, ax=axes[row, column], shrink=0.8, label="relative velocity")
    axis = axes[:, 5][0]
    axis.semilogy(data_history, "o-", label="data MSE")
    axis.semilogy(total_history, "o-", label="total objective")
    axis.set(title="inversion progress", xlabel="iteration", ylabel="loss")
    axis.grid(alpha=0.3)
    axis.legend()
    axes[1, 5].axis("off")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--spacing", type=float, default=1.0, help="local forward-grid spacing in km")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--event-count", type=int, default=30)
    parser.add_argument("--checker-amplitude", type=float, default=0.05)
    parser.add_argument("--checker-lon-nodes", type=int, default=12)
    parser.add_argument("--checker-lat-nodes", type=int, default=12)
    parser.add_argument("--checker-depth-nodes", type=int, default=3)
    parser.add_argument("--background-model", type=Path, default=DEFAULT_BACKGROUND_MODEL)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--validate-ddp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.spacing <= 0 or args.checker_amplitude <= 0 or args.checker_amplitude >= 1:
        raise ValueError("spacing must be positive and checker amplitude must be in (0, 1)")
    if min(args.checker_lon_nodes, args.checker_lat_nodes, args.checker_depth_nodes, args.event_count) < 1:
        raise ValueError("event count and checkerboard block sizes must be positive")
    distributed, rank, world_size = distributed_context()
    try:
        if not args.skip_prepare and rank == 0:
            args.data_dir.mkdir(parents=True, exist_ok=True)
            _, true = generate_models(
                args.background_model,
                args.data_dir,
                args.checker_amplitude,
                args.checker_lon_nodes,
                args.checker_lat_nodes,
                args.checker_depth_nodes,
            )
            stations = generate_stations(args.data_dir)
            events = generate_events(args.data_dir, args.event_count, args.seed, true)
            true = torch.load(args.data_dir / "model_true.pt", weights_only=True)
            generate_picks(args.data_dir, stations, events, true, args.spacing)
            print(f"prepared {len(events)} events and {2 * len(events) * len(stations)} P/S picks in {args.data_dir}")
        if distributed:
            dist.barrier()
        initial, true, stations, events, picks = load_data(args.data_dir)
        model_for_grids = VelocityModel(**initial, trainable=False)
        groups = prepare_pick_groups(stations, events, picks, model_for_grids, args.spacing)
        local_groups = groups[rank::world_size]
        if rank == 0:
            print(f"world_size={world_size}; station groups per rank={[len(groups[r::world_size]) for r in range(world_size)]}")
        model, data_history, total_history = run_inversion(
            initial, groups, local_groups, args, distributed, rank, world_size
        )
        if distributed:
            dist.barrier()
        if rank == 0:
            args.results_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"lon": model.lon, "lat": model.lat, "depth": model.depth, "vp": model.vp.detach(), "vs": model.vs.detach()},
                args.results_dir / "model_inverted.pt",
            )
            plot_results(true, initial, model, data_history, total_history, args.results_dir / "inversion_progress.png")
            print(f"saved final model and plot to {args.results_dir}")
        if distributed:
            dist.barrier()
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
