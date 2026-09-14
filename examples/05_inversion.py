"""Invert direct global Vp/Vs from fixed catalog arrival times."""

from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt

from adtomo import ForwardGrid, VelocityModel, predict_travel_times

DATA = Path("data")
RESULTS = Path("results")


def prepare_pick_groups(stations, events, picks, model):
    events_by_id = events.set_index("event_id")
    stations_by_id = stations.set_index("station_id")

    groups = []
    for station_id, station_picks in picks.groupby("station_id", sort=False):
        station = stations_by_id.loc[station_id]
        station_event_ids = pd.unique(station_picks.event_id)
        station_events = events_by_id.loc[station_event_ids]
        station_lonlatdepth = torch.tensor([station.longitude, station.latitude, station.depth_km], dtype=torch.float64)
        event_lonlatdepth = torch.tensor(
            station_events[["longitude", "latitude", "depth_km"]].values, dtype=torch.float64
        )
        grid = ForwardGrid(station_lonlatdepth, event_lonlatdepth, model, spacing=5.0)

        for phase, phase_picks in station_picks.groupby("phase_type", sort=False):
            grid_event_indices = torch.tensor(pd.Index(station_event_ids).get_indexer(phase_picks.event_id), dtype=torch.long)
            catalog_event_time = pd.to_datetime(phase_picks.event_id.map(events_by_id.event_time))
            phase_dt = torch.tensor(
                (pd.to_datetime(phase_picks.phase_time) - catalog_event_time).dt.total_seconds().to_numpy(),
                dtype=torch.float64,
            )
            groups.append(
                {
                    "grid": grid,
                    "phase": phase,
                    "grid_event_indices": grid_event_indices,
                    "observed_phase_dt": phase_dt,
                }
            )
    return groups


def plot_progress(true, initial, model, loss_history, path):
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
    axis.semilogy(loss_history, "o-", color="tab:blue")
    axis.set_title("Arrival-time inversion")
    axis.set_xlabel("Adam iteration")
    axis.set_ylabel("phase-time MSE (s²)")
    axis.grid(alpha=0.3)
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

optimizer = torch.optim.Adam([model.vp, model.vs], lr=0.03)
loss_history = []
for iteration in range(31):
    optimizer.zero_grad()
    residuals = []
    for group in groups:
        predicted_phase_dt = predict_travel_times(
            model,
            group["grid"],
            group["phase"],
            event_indices=group["grid_event_indices"],
        )
        residuals.append(predicted_phase_dt - group["observed_phase_dt"])
    loss = torch.cat(residuals).square().mean()
    loss_history.append(loss.item())
    if iteration == 0:
        initial_loss = loss.item()
    if iteration < 30:
        loss.backward()
        optimizer.step()
    if iteration % 5 == 0 or iteration == 30:
        print(f"iteration {iteration:02d} phase-time MSE {loss.item():.6f}")

RESULTS.mkdir(exist_ok=True)
torch.save(
    {"lon": model.lon, "lat": model.lat, "depth": model.depth, "vp": model.vp.detach(), "vs": model.vs.detach()},
    RESULTS / "model_inverted.pt",
)
plot_progress(true, initial, model, loss_history, RESULTS / "inversion_progress.png")
print(f"saved final model to {RESULTS}; loss {initial_loss:.6f} -> {loss.item():.6f}")
