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


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODEL_HORIZONTAL_SPACING_DEG = 0.1
MODEL_DEPTH_SPACING_KM = 5.0
FORWARD_SPACING_KM = 5.0
ITERATIONS = int(os.environ.get("ADTOMO_ITERATIONS", "100"))
LAMBDA_VP = float(os.environ.get("ADTOMO_LAMBDA_VP", "0.1"))
LAMBDA_VS = float(os.environ.get("ADTOMO_LAMBDA_VS", "0.1"))


def regular_axis(lower, upper, spacing):
    start = math.floor(lower / spacing) * spacing
    stop = math.ceil(upper / spacing) * spacing
    return torch.arange(start, stop + 0.5 * spacing, spacing, dtype=torch.float64)


def distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return 0, 1
    dist.init_process_group("gloo")
    return dist.get_rank(), world_size


def initial_model(events, stations):
    coordinates = pd.concat(
        [
            events[["longitude", "latitude", "depth_km"]],
            stations[["longitude", "latitude", "depth_km"]],
        ]
    )
    lon = regular_axis(
        coordinates.longitude.min() - 1.0, coordinates.longitude.max() + 1.0, MODEL_HORIZONTAL_SPACING_DEG
    )
    lat = regular_axis(
        coordinates.latitude.min() - 1.0, coordinates.latitude.max() + 1.0, MODEL_HORIZONTAL_SPACING_DEG
    )
    depth = regular_axis(-50.0, coordinates.depth_km.max() + 30.0, MODEL_DEPTH_SPACING_KM)
    depth_grid, _, _ = torch.meshgrid(depth, lat, lon, indexing="ij")
    vp = 5.5 + 0.03 * depth_grid.clamp_min(0.0)
    return VelocityModel(lon, lat, depth, vp, vp / 1.73, trainable=True)


def prepare_station_groups(events, stations, picks, model, rank, world_size):
    events_by_id = events.set_index("event_id")
    stations_by_id = stations.set_index("station_id")
    station_ids = sorted(picks.station_id.unique())
    local_station_ids = station_ids[rank::world_size]
    groups = []
    for station_id in local_station_ids:
        station_picks = picks[picks.station_id == station_id]
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
    return groups


def plot_result(model, initial_vp, initial_vs, loss_history, path):
    depth_index = int(torch.argmin((model.depth - 10.0).abs()))
    extent = [model.lon[0].item(), model.lon[-1].item(), model.lat[0].item(), model.lat[-1].item()]
    fields = [
        ("Vp initial", initial_vp[depth_index]),
        ("Vp inverted", model.vp.detach()[depth_index]),
        ("Vp change", model.vp.detach()[depth_index] - initial_vp[depth_index]),
        ("Vs initial", initial_vs[depth_index]),
        ("Vs inverted", model.vs.detach()[depth_index]),
        ("Vs change", model.vs.detach()[depth_index] - initial_vs[depth_index]),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
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
    axes[0, 3].semilogy(loss_history, "o-", color="tab:blue")
    axes[0, 3].set(title="Full-catalog DDP inversion", xlabel="Adam iteration", ylabel="phase-time MSE (s²)")
    axes[0, 3].grid(alpha=0.3)
    axes[1, 3].axis("off")
    figure.suptitle(f"Direct two-grid inversion at {model.depth[depth_index].item():.1f} km depth")
    figure.savefig(path, dpi=180)
    plt.close(figure)


rank, world_size = distributed_context()
events = pd.read_csv(ROOT / "events.csv", dtype={"event_id": str})
stations = pd.read_csv(ROOT / "stations.csv", dtype={"station_id": str})
picks = pd.read_csv(ROOT / "picks.csv", dtype={"event_id": str, "station_id": str})
if set(picks.event_id) - set(events.event_id) or set(picks.station_id) - set(stations.station_id):
    raise ValueError("picks reference an unknown catalog event or station")

model = initial_model(events, stations)
initial_vp, initial_vs = model.vp.detach().clone(), model.vs.detach().clone()
tomography = Tomography(model, lambda_vp=LAMBDA_VP, lambda_vs=LAMBDA_VS)
ddp_tomography = DistributedDataParallel(tomography, find_unused_parameters=False) if world_size > 1 else tomography
station_groups = prepare_station_groups(events, stations, picks, model, rank, world_size)
local_pick_count = sum(len(observed_phase_dt) for _, phase_groups in station_groups for _, _, observed_phase_dt in phase_groups)

if rank == 0:
    print(
        f"using all {len(events)} events, {len(stations)} stations, and {len(picks)} picks "
        f"({(picks.phase_type == 'P').sum()} P, {(picks.phase_type == 'S').sum()} S)"
    )
    print(f"DDP ranks={world_size}; global model shape (depth, latitude, longitude)={tuple(model.vp.shape)}")

optimizer = torch.optim.Adam([p for p in tomography.parameters() if p.requires_grad], lr=0.01)
loss_history = []
for iteration in range(ITERATIONS):
    optimizer.zero_grad()
    loss = ddp_tomography(station_groups, world_size * local_pick_count / len(picks))
    if iteration < ITERATIONS - 1:
        loss.backward()
    if iteration < ITERATIONS - 1:
        optimizer.step()
    data_sum = tomography.data_loss * local_pick_count
    if world_size > 1:
        dist.all_reduce(data_sum, op=dist.ReduceOp.SUM)
    data_loss = (data_sum / len(picks)).item()
    reg_vp = tomography.reg_vp.item()
    reg_vs = tomography.reg_vs.item()
    total_loss = data_loss + LAMBDA_VP * reg_vp + LAMBDA_VS * reg_vs
    loss_history.append(data_loss)
    if rank == 0 and (iteration % 5 == 0 or iteration == ITERATIONS - 1):
        print(
            f"iteration {iteration:02d} total={total_loss:.6f} data={data_loss:.6f} "
            f"reg_vp={reg_vp:.6f} reg_vs={reg_vs:.6f}"
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
            "model_horizontal_spacing_deg": MODEL_HORIZONTAL_SPACING_DEG,
            "model_depth_spacing_km": MODEL_DEPTH_SPACING_KM,
            "forward_spacing_km": FORWARD_SPACING_KM,
            "lambda_vp": LAMBDA_VP,
            "lambda_vs": LAMBDA_VS,
        },
        RESULTS / "model_inverted_old_ddp.pt",
    )
    plot_result(model, initial_vp, initial_vs, loss_history, RESULTS / "inversion_progress_old_ddp.png")
    print(f"saved final model and figure to {RESULTS}; loss {loss_history[0]:.6f} -> {loss_history[-1]:.6f}")

if world_size > 1:
    dist.barrier()
    dist.destroy_process_group()
