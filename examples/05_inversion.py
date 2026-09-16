"""Invert direct global Vp/Vs from fixed catalog arrival times."""

from pathlib import Path
import os

import pandas as pd
import torch
import matplotlib.pyplot as plt

from adtomo import ForwardGrid, Tomography, VelocityModel

DATA = Path("data")
RESULTS = Path("results")
LAMBDA_VP = float(os.environ.get("ADTOMO_LAMBDA_VP", "0.0"))
LAMBDA_VS = float(os.environ.get("ADTOMO_LAMBDA_VS", "0.0"))
ALPHA_VP = float(os.environ.get("ADTOMO_ALPHA_VP", "0.0"))
ALPHA_VS = float(os.environ.get("ADTOMO_ALPHA_VS", "0.0"))


def prepare_pick_groups(stations, events, picks, model):
    events_by_id = events.set_index("event_id")
    stations_by_id = stations.set_index("station_id")

    groups = []
    for station_id, station_picks in picks.groupby("station_id", sort=False):
        station = stations_by_id.loc[station_id]
        station_event_ids = pd.unique(station_picks.event_id)
        station_events = events_by_id.loc[station_event_ids]
        station_spherical = torch.tensor([station.longitude, station.latitude, station.depth_km], dtype=torch.float64)
        events_spherical = torch.tensor(
            station_events[["longitude", "latitude", "depth_km"]].values, dtype=torch.float64
        )
        grid = ForwardGrid(station_spherical, events_spherical, model, spacing=5.0)

        phase_groups = []
        for phase, phase_picks in station_picks.groupby("phase_type", sort=False):
            grid_event_indices = torch.tensor(pd.Index(station_event_ids).get_indexer(phase_picks.event_id), dtype=torch.long)
            catalog_event_time = pd.to_datetime(phase_picks.event_id.map(events_by_id.event_time))
            phase_dt = torch.tensor(
                (pd.to_datetime(phase_picks.phase_time) - catalog_event_time).dt.total_seconds().to_numpy(),
                dtype=torch.float64,
            )
            phase_groups.append((phase, grid_event_indices, phase_dt))
        groups.append((grid, phase_groups))
    return groups


def plot_progress(true, initial, model, data_loss_history, total_loss_history, path):
    depth_index = int(torch.argmin((model.depth - 15.0).abs()))
    depth_km = model.depth[depth_index].item()
    extent = [model.lon[0].item(), model.lon[-1].item(), model.lat[0].item(), model.lat[-1].item()]
    fields = [
        ("Vp true", true["vp"][depth_index]),
        ("Vp initial", initial["vp"][depth_index]),
        ("Vp inverted", model.vp.detach()[depth_index]),
        ("Vs true", true["vs"][depth_index]),
        ("Vs initial", initial["vs"][depth_index]),
        ("Vs inverted", model.vs.detach()[depth_index]),
    ]
    figure = plt.figure(figsize=(15, 7), constrained_layout=True)
    layout = figure.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.15])
    for row, phase in enumerate(("Vp", "Vs")):
        phase_fields = [field for title, field in fields if title.startswith(phase)]
        vmin = min(field.min().item() for field in phase_fields)
        vmax = max(field.max().item() for field in phase_fields)
        for column, (title, field) in enumerate((item for item in fields if item[0].startswith(phase))):
            axis = figure.add_subplot(layout[row, column])
            image = axis.imshow(field, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
            axis.set_title(title)
            axis.set_xlabel("longitude (deg)")
            axis.set_ylabel("latitude (deg)")
            figure.colorbar(image, ax=axis, shrink=0.8, label="km/s")
    axis = figure.add_subplot(layout[:, 3])
    axis.semilogy(data_loss_history, "o-", color="tab:blue", label="Data MSE")
    axis.semilogy(total_loss_history, "o-", color="tab:orange", label="Total objective")
    axis.set_title("Arrival-time inversion")
    axis.set_xlabel("Adam iteration")
    axis.set_ylabel("Objective value")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.suptitle(f"Two-grid spherical inversion progress at depth {depth_km:.1f} km")
    figure.savefig(path, dpi=180)
    plt.close(figure)


initial = torch.load(DATA / "model_initial.pt", weights_only=True)
true = torch.load(DATA / "model_true.pt", weights_only=True)
model = VelocityModel(**initial, trainable=True)
stations = pd.read_csv(DATA / "stations.csv", dtype={"station_id": str})
events = pd.read_csv(DATA / "events.csv", dtype={"event_id": str})
picks = pd.read_csv(DATA / "picks.csv", dtype={"event_id": str, "station_id": str})
groups = prepare_pick_groups(stations, events, picks, model)

tomography = Tomography(
    model,
    lambda_vp=LAMBDA_VP,
    lambda_vs=LAMBDA_VS,
    alpha_vp=ALPHA_VP,
    alpha_vs=ALPHA_VS,
)
optimizer = torch.optim.Adam([p for p in tomography.parameters() if p.requires_grad], lr=0.03)
data_loss_history = []
total_loss_history = []
for iteration in range(31):
    optimizer.zero_grad()
    loss = tomography(groups)
    data_loss_history.append(tomography.data_loss.item())
    total_loss_history.append(tomography.total_loss.item())
    if iteration == 0:
        initial_loss = tomography.data_loss.item()
    if iteration < 30:
        loss.backward()
        optimizer.step()
    if iteration % 5 == 0 or iteration == 30:
        print(
            f"iteration {iteration:02d} total={loss.item():.6f} data={tomography.data_loss.item():.6f} "
            f"smooth_vp={tomography.smooth_vp.item():.6f} smooth_vs={tomography.smooth_vs.item():.6f} "
            f"damp_vp={tomography.damp_vp.item():.6f} damp_vs={tomography.damp_vs.item():.6f}"
        )

RESULTS.mkdir(exist_ok=True)
torch.save(
    {"lon": model.lon, "lat": model.lat, "depth": model.depth, "vp": model.vp.detach(), "vs": model.vs.detach()},
    RESULTS / "model_inverted.pt",
)
plot_progress(true, initial, model, data_loss_history, total_loss_history, RESULTS / "inversion_progress.png")
print(
    f"saved final model to {RESULTS}; data MSE {initial_loss:.6f} -> {tomography.data_loss.item():.6f}; "
    f"minimum Vp/Vs {model.vp.detach().min().item():.6f}/{model.vs.detach().min().item():.6f} km/s"
)
