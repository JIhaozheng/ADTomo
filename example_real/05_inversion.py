"""DDP inversion of every supplied real P and S catalog arrival.

Launch with one process per CPU worker, for example:

    torchrun --standalone --nproc_per_node=8 05_inversion.py

Each rank follows the original ADTomo parallel pattern: it evaluates all of
its assigned stations in one DDP forward call and synchronizes gradients once
per Adam step.
"""

from pathlib import Path
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from adtomo import ForwardGrid, Tomography, VelocityModel
from adtomo.grid import ecef_to_local, local_basis, spherical_to_ecef


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FORWARD_SPACING_KM = 2.0
ITERATIONS = int(os.environ.get("ADTOMO_ITERATIONS", "100"))
EVENT_COUNT = int(os.environ.get("ADTOMO_EVENT_COUNT", "1000"))
EVENT_SEED = int(os.environ.get("ADTOMO_EVENT_SEED", "20260914"))
LAMBDA_VP = float(os.environ.get("ADTOMO_LAMBDA_VP", "0.1"))
LAMBDA_VS = float(os.environ.get("ADTOMO_LAMBDA_VS", "0.1"))
ALPHA_VP = float(os.environ.get("ADTOMO_ALPHA_VP", "0.0"))
ALPHA_VS = float(os.environ.get("ADTOMO_ALPHA_VS", "0.0"))


def distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return 0, 1
    dist.init_process_group("gloo")
    return dist.get_rank(), world_size


def configure_cpu_threads(world_size):
    available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    threads = int(os.environ.get("ADTOMO_THREADS_PER_RANK", max(1, available_cpus // world_size)))
    if threads < 1:
        raise ValueError("ADTOMO_THREADS_PER_RANK must be positive")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    return threads


def station_work_weights(events, stations, picks):
    events_by_id = events.set_index("event_id")
    stations_by_id = stations.set_index("station_id")
    weights = {}
    for station_id, station_picks in picks.groupby("station_id", sort=False):
        station = stations_by_id.loc[station_id]
        station_lonlatdepth = torch.tensor(
            [station.longitude, station.latitude, station.depth_km], dtype=torch.float64
        )
        event_lonlatdepth = torch.tensor(
            events_by_id.loc[pd.unique(station_picks.event_id), ["longitude", "latitude", "depth_km"]].values,
            dtype=torch.float64,
        )
        station_ecef = spherical_to_ecef(*station_lonlatdepth)
        event_local = ecef_to_local(
            spherical_to_ecef(event_lonlatdepth[:, 0], event_lonlatdepth[:, 1], event_lonlatdepth[:, 2]),
            station_ecef,
            local_basis(station_lonlatdepth[0], station_lonlatdepth[1]),
        )
        points = torch.cat([torch.zeros((1, 3), dtype=torch.float64), event_local])
        low = points.amin(dim=0) - 2.0 * FORWARD_SPACING_KM
        high = points.amax(dim=0) + 2.0 * FORWARD_SPACING_KM
        nxyz = [max(2, math.ceil(float((high[index] - low[index]) / FORWARD_SPACING_KM)) + 1) for index in range(3)]
        weights[station_id] = math.prod(nxyz) * station_picks.phase_type.nunique()
    return weights


def balanced_station_ids(picks, work_weights, world_size):
    pick_counts = picks.groupby("station_id", sort=False).size()
    station_ids_by_rank = [[] for _ in range(world_size)]
    rank_pick_counts = [0] * world_size
    rank_work_weights = [0] * world_size
    for station_id in sorted(pick_counts.index, key=lambda station_id: (-work_weights[station_id], station_id)):
        rank = min(range(world_size), key=lambda index: rank_work_weights[index])
        station_ids_by_rank[rank].append(station_id)
        rank_pick_counts[rank] += pick_counts[station_id]
        rank_work_weights[rank] += work_weights[station_id]
    return station_ids_by_rank, rank_pick_counts, rank_work_weights


def prepare_station_groups(events, stations, picks, work_weights, model, rank, world_size):
    events_by_id = events.set_index("event_id")
    stations_by_id = stations.set_index("station_id")
    station_picks_by_id = {station_id: station_picks for station_id, station_picks in picks.groupby("station_id", sort=False)}
    station_ids_by_rank, rank_pick_counts, rank_work_weights = balanced_station_ids(picks, work_weights, world_size)
    groups = []
    for station_id in station_ids_by_rank[rank]:
        station_picks = station_picks_by_id[station_id]
        station = stations_by_id.loc[station_id]
        station_event_ids = pd.unique(station_picks.event_id)
        station_events = events_by_id.loc[station_event_ids]
        grid = ForwardGrid(
            torch.tensor([station.longitude, station.latitude, station.depth_km], dtype=torch.float64),
            torch.tensor(station_events[["longitude", "latitude", "depth_km"]].values, dtype=torch.float64),
            model,
            spacing=FORWARD_SPACING_KM,
        )
        phase_groups = []
        for phase in ("P", "S"):
            phase_picks = station_picks[station_picks.phase_type == phase]
            if phase_picks.empty:
                continue
            grid_event_indices = torch.tensor(
                pd.Index(station_event_ids).get_indexer(phase_picks.event_id), dtype=torch.long
            )
            event_time = pd.to_datetime(phase_picks.event_id.map(events_by_id.event_time))
            observed_phase_dt = torch.tensor(
                (pd.to_datetime(phase_picks.phase_time) - event_time).dt.total_seconds().to_numpy(),
                dtype=torch.float64,
            )
            phase_groups.append((phase, grid_event_indices, observed_phase_dt))
        groups.append((grid, phase_groups))
    return groups, rank_pick_counts, rank_work_weights


def plot_result(model, initial_vp, initial_vs, loss_history, results):
    extent = [model.lon[0].item(), model.lon[-1].item(), model.lat[0].item(), model.lat[-1].item()]
    slices = results / "depth_slices"
    slices.mkdir(exist_ok=True)
    for depth_index, depth_km in enumerate(model.depth.tolist()):
        fields = [
            ("Vp initial", initial_vp[depth_index]),
            ("Vp inverted", model.vp.detach()[depth_index]),
            ("Vp change", model.vp.detach()[depth_index] - initial_vp[depth_index]),
            ("Vs initial", initial_vs[depth_index]),
            ("Vs inverted", model.vs.detach()[depth_index]),
            ("Vs change", model.vs.detach()[depth_index] - initial_vs[depth_index]),
        ]
        figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        for row in range(2):
            phase_fields = fields[row * 3 : (row + 1) * 3]
            speed_limits = (
                min(field.min().item() for _, field in phase_fields[:2]),
                max(field.max().item() for _, field in phase_fields[:2]),
            )
            change_limit = max(abs(phase_fields[2][1].min().item()), abs(phase_fields[2][1].max().item()))
            for column, (title, field) in enumerate(phase_fields):
                kwargs = {"cmap": "viridis", "vmin": speed_limits[0], "vmax": speed_limits[1]}
                label = "km/s"
                if column == 2:
                    kwargs = {"cmap": "seismic", "vmin": -change_limit, "vmax": change_limit}
                    label = "km/s change"
                image = axes[row, column].imshow(field, origin="lower", extent=extent, **kwargs)
                axes[row, column].set(title=title, xlabel="longitude (deg)", ylabel="latitude (deg)")
                figure.colorbar(image, ax=axes[row, column], shrink=0.82, label=label)
        figure.suptitle(f"Direct two-grid inversion at {depth_km:.1f} km depth")
        figure.savefig(slices / f"depth_{depth_km:05.1f}_km.png", dpi=180)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    axis.semilogy(loss_history, "o-", color="tab:blue")
    axis.set(title="Full-catalog DDP inversion", xlabel="Adam iteration", ylabel="phase-time MSE (s²)")
    axis.grid(alpha=0.3)
    figure.savefig(results / "inversion_progress.png", dpi=180)
    plt.close(figure)


rank, world_size = distributed_context()
threads_per_rank = configure_cpu_threads(world_size)
events = pd.read_csv(DATA / "events.csv", dtype={"event_id": str})
stations = pd.read_csv(DATA / "stations.csv", dtype={"station_id": str})
picks = pd.read_csv(DATA / "picks.csv", dtype={"event_id": str, "station_id": str})
if set(picks.event_id) - set(events.event_id) or set(picks.station_id) - set(stations.station_id):
    raise ValueError("picks reference an unknown catalog event or station")
if not 1 <= EVENT_COUNT <= len(events):
    raise ValueError(f"ADTOMO_EVENT_COUNT must be in [1, {len(events)}], got {EVENT_COUNT}")
events = events.sample(n=EVENT_COUNT, random_state=EVENT_SEED).reset_index(drop=True)
picks = picks[picks.event_id.isin(events.event_id)].copy()
stations = stations[stations.station_id.isin(picks.station_id)].copy()

model = VelocityModel(**torch.load(DATA / "model_initial.pt", weights_only=True), trainable=True)
initial_vp, initial_vs = model.vp.detach().clone(), model.vs.detach().clone()
tomography = Tomography(
    model,
    lambda_vp=LAMBDA_VP,
    lambda_vs=LAMBDA_VS,
    alpha_vp=ALPHA_VP,
    alpha_vs=ALPHA_VS,
)
ddp_tomography = DistributedDataParallel(tomography, find_unused_parameters=False) if world_size > 1 else tomography
work_weights = station_work_weights(events, stations, picks)
station_groups, rank_pick_counts, rank_work_weights = prepare_station_groups(
    events, stations, picks, work_weights, model, rank, world_size
)
local_pick_count = sum(len(observed_phase_dt) for _, phase_groups in station_groups for _, _, observed_phase_dt in phase_groups)

if rank == 0:
    print(
        f"using {len(events)} random events (seed={EVENT_SEED}), {len(stations)} stations, and {len(picks)} picks "
        f"({(picks.phase_type == 'P').sum()} P, {(picks.phase_type == 'S').sum()} S)"
    )
    print(f"DDP ranks={world_size}; global model shape (depth, latitude, longitude)={tuple(model.vp.shape)}")
    print(f"CPU threads per rank={threads_per_rank}; balanced picks per rank={rank_pick_counts}")
    print(f"balanced forward cells × phases per rank={rank_work_weights}")

optimizer = torch.optim.Adam([p for p in tomography.parameters() if p.requires_grad], lr=0.01)
loss_history = []
for iteration in range(ITERATIONS):
    optimizer.zero_grad()
    loss = ddp_tomography(station_groups) * (world_size * local_pick_count / len(picks))
    if iteration < ITERATIONS - 1:
        loss.backward()
    if iteration < ITERATIONS - 1:
        optimizer.step()
    data_sum = tomography.data_loss * local_pick_count
    if world_size > 1:
        dist.all_reduce(data_sum, op=dist.ReduceOp.SUM)
    data_loss = (data_sum / len(picks)).item()
    smooth_vp = tomography.smooth_vp.item()
    smooth_vs = tomography.smooth_vs.item()
    damp_vp = tomography.damp_vp.item()
    damp_vs = tomography.damp_vs.item()
    total_loss = data_loss + LAMBDA_VP * smooth_vp + LAMBDA_VS * smooth_vs + ALPHA_VP * damp_vp + ALPHA_VS * damp_vs
    loss_history.append(data_loss)
    if rank == 0 and (iteration % 5 == 0 or iteration == ITERATIONS - 1):
        print(
            f"iteration {iteration:02d} total={total_loss:.6f} data={data_loss:.6f} "
            f"smooth_vp={smooth_vp:.6f} smooth_vs={smooth_vs:.6f} "
            f"damp_vp={damp_vp:.6f} damp_vs={damp_vs:.6f}"
        )

if rank == 0:
    RESULTS.mkdir(exist_ok=True)
    torch.save(
        {
            "lon": model.lon,
            "lat": model.lat,
            "depth": model.depth,
            "vp": model.vp.detach(),
            "vs": model.vs.detach(),
            "event_count": len(events),
            "station_count": len(stations),
            "pick_count": len(picks),
            "world_size": world_size,
            "loss_history": loss_history,
            "forward_spacing_km": FORWARD_SPACING_KM,
            "lambda_vp": LAMBDA_VP,
            "lambda_vs": LAMBDA_VS,
            "alpha_vp": ALPHA_VP,
            "alpha_vs": ALPHA_VS,
        },
        RESULTS / "model_inverted.pt",
    )
    plot_result(model, initial_vp, initial_vs, loss_history, RESULTS)
    print(f"saved final model and {len(model.depth)} depth figures to {RESULTS}; loss {loss_history[0]:.6f} -> {loss_history[-1]:.6f}")

if world_size > 1:
    dist.barrier()
    dist.destroy_process_group()
