"""Run the synthetic station-group inversion with CPU DistributedDataParallel.

Launch from ``examples`` with, for example,
``torchrun --standalone --nproc_per_node=2 parallel/inversion_ddp.py``.
Use ``--validate`` to compare one DDP SGD update against the serial objective.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from adtomo import ForwardGrid, Tomography, VelocityModel, predict_travel_times, smoothness


DATA = Path(__file__).resolve().parents[1] / "data"


def prepare_pick_groups(stations, events, picks, model):
    events_by_id = events.set_index("event_id")
    stations_by_id = stations.set_index("station_id")
    groups = []
    for station_id, station_picks in picks.groupby("station_id", sort=False):
        station = stations_by_id.loc[station_id]
        event_ids = pd.unique(station_picks.event_id)
        station_spherical = torch.tensor([station.longitude, station.latitude, station.depth_km], dtype=torch.float64)
        events_spherical = torch.tensor(
            events_by_id.loc[event_ids, ["longitude", "latitude", "depth_km"]].values, dtype=torch.float64
        )
        grid = ForwardGrid(station_spherical, events_spherical, model, spacing=5.0)
        phase_groups = []
        for phase, phase_picks in station_picks.groupby("phase_type", sort=False):
            event_indices = torch.tensor(pd.Index(event_ids).get_indexer(phase_picks.event_id), dtype=torch.long)
            origin_time = pd.to_datetime(phase_picks.event_id.map(events_by_id.event_time))
            observed = torch.tensor(
                (pd.to_datetime(phase_picks.phase_time) - origin_time).dt.total_seconds().to_numpy(), dtype=torch.float64
            )
            phase_groups.append((phase, event_indices, observed))
        groups.append((grid, phase_groups))
    return groups


def squared_residual_sum(model, groups):
    residuals = []
    for grid, phase_groups in groups:
        for phase, event_indices, observed in phase_groups:
            residuals.append(predict_travel_times(model, grid, phase, event_indices) - observed)
    return torch.cat(residuals).square().sum()


def observation_count(groups):
    return sum(len(observed) for _, phase_groups in groups for _, _, observed in phase_groups)


class StationGroupObjective(nn.Module):
    """DDP-safe objective whose averaged gradient equals the serial objective."""

    def __init__(self, tomography, world_size, total_observations):
        super().__init__()
        self.tomography = tomography
        self.world_size = world_size
        self.total_observations = total_observations

    def forward(self, local_groups):
        model = self.tomography.model
        data_loss = self.world_size * squared_residual_sum(model, local_groups) / self.total_observations
        dvp = model.vp - self.tomography.vp0
        dvs = model.vs - self.tomography.vs0
        regularization = (
            self.tomography.lambda_vp * smoothness(dvp, model.lon, model.lat, model.depth)
            + self.tomography.lambda_vs * smoothness(dvs, model.lon, model.lat, model.depth)
            + self.tomography.alpha_vp * dvp.square().mean()
            + self.tomography.alpha_vs * dvs.square().mean()
        )
        return data_loss + regularization


def global_metrics(objective, local_groups):
    """Return serial-equivalent scalar losses after a DDP step."""
    local_sum = squared_residual_sum(objective.tomography.model, local_groups).detach()
    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
    model = objective.tomography.model
    dvp = model.vp - objective.tomography.vp0
    dvs = model.vs - objective.tomography.vs0
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(0)

    initial = torch.load(DATA / "model_initial.pt", weights_only=True)
    model = VelocityModel(**initial, trainable=True)
    stations = pd.read_csv(DATA / "stations.csv", dtype={"station_id": str})
    events = pd.read_csv(DATA / "events.csv", dtype={"event_id": str})
    picks = pd.read_csv(DATA / "picks.csv", dtype={"event_id": str, "station_id": str})
    groups = prepare_pick_groups(stations, events, picks, model)
    local_groups = groups[rank::world_size]
    total_observations = observation_count(groups)
    if not local_groups:
        raise ValueError("world_size cannot exceed the number of station groups")

    tomography = Tomography(model)
    objective = StationGroupObjective(tomography, world_size, total_observations)
    distributed = DistributedDataParallel(objective)
    optimizer_type = torch.optim.SGD if args.validate else torch.optim.Adam
    optimizer = optimizer_type(distributed.parameters(), lr=args.learning_rate)

    reference = None
    if args.validate and rank == 0:
        reference_model = VelocityModel(**initial, trainable=True)
        reference = Tomography(reference_model)

    iterations = 1 if args.validate else args.iterations
    for iteration in range(iterations):
        optimizer.zero_grad()
        loss = distributed(local_groups)
        loss.backward()
        global_data_loss, global_total_loss = global_metrics(objective, local_groups)
        if reference is not None:
            reference_loss = reference(groups)
            reference_loss.backward()
            assert torch.allclose(global_data_loss, reference.data_loss, atol=1e-12)
            assert torch.allclose(global_total_loss, reference.total_loss, atol=1e-12)
            vp_error = relative_error(model.vp.grad, reference.model.vp.grad).item()
            vs_error = relative_error(model.vs.grad, reference.model.vs.grad).item()
            assert vp_error < 1e-11 and vs_error < 1e-11, (vp_error, vs_error)
        optimizer.step()

    data_loss, total_loss = global_metrics(objective, local_groups)
    if reference is not None:
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=args.learning_rate)
        reference_optimizer.step()
        vp_error = relative_error(model.vp.detach(), reference.model.vp.detach()).item()
        vs_error = relative_error(model.vs.detach(), reference.model.vs.detach()).item()
        assert vp_error < 1e-11 and vs_error < 1e-11, (vp_error, vs_error)
        with torch.no_grad():
            reference(groups)
        assert torch.allclose(data_loss, reference.data_loss, atol=1e-12)
        assert torch.allclose(total_loss, reference.total_loss, atol=1e-12)
        print("DDP validation passed: serial objective, gradients, and one SGD update agree.")
    if rank == 0:
        print(
            f"world_size={world_size} iterations={iterations} "
            f"data MSE={data_loss.item():.6f} total={total_loss.item():.6f}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
