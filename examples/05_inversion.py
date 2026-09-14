"""Invert direct global Vp/Vs from fixed catalog arrival times."""

from pathlib import Path
import sys

import pandas as pd
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adtomo import ForwardGrid, VelocityModel, predict_phase_times

DATA = Path(__file__).resolve().parent / "data"
RESULTS = Path(__file__).resolve().parent / "results"


def main():
    initial = torch.load(DATA / "model_initial.pt", weights_only=True)
    true = torch.load(DATA / "model_true.pt", weights_only=True)
    model = VelocityModel(**initial, trainable=True)
    stations = pd.read_csv(DATA / "stations.csv", dtype={"station_id": str})
    events = pd.read_csv(DATA / "events.csv", dtype={"event_id": str})
    picks = pd.read_csv(DATA / "picks.csv", dtype={"event_id": str, "station_id": str})
    event_xyz = torch.tensor(events[["longitude", "latitude", "depth_km"]].values, dtype=torch.float64)
    event_number = {event_id: i for i, event_id in enumerate(events.event_id)}
    event_time = pd.Series(pd.to_datetime(events.event_time).values, index=events.event_id)
    grids = {}
    for station in stations.itertuples(index=False):
        grids[station.station_id] = ForwardGrid(
            torch.tensor([station.longitude, station.latitude, station.depth_km], dtype=torch.float64),
            event_xyz,
            model,
            spacing=5.0,
        )

    groups = []
    for (station_id, phase), rows in picks.groupby(["station_id", "phase_type"], sort=False):
        event_indices = torch.tensor([event_number[event_id] for event_id in rows.event_id], dtype=torch.long)
        catalog_time = pd.to_datetime(rows.event_id.map(event_time))
        phase_dt = (pd.to_datetime(rows.phase_time) - catalog_time).dt.total_seconds().to_numpy()
        groups.append((grids[station_id], phase, event_indices, torch.tensor(phase_dt, dtype=torch.float64)))

    optimizer = torch.optim.Adam([model.vp, model.vs], lr=0.03)
    loss_history = []
    for iteration in range(31):
        optimizer.zero_grad()
        residuals = []
        for grid, phase, event_indices, observed in groups:
            predicted = predict_phase_times(
                model, grid, phase, torch.zeros(len(event_indices), dtype=torch.float64), event_indices=event_indices
            )
            residuals.append(predicted - observed)
        loss = torch.cat(residuals).square().mean()
        loss_history.append(loss.item())
        if iteration == 0:
            initial_loss = loss.item()
        if iteration < 30:
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.vp.clamp_(min=1.0)
                model.vs.clamp_(min=1.0)
        if iteration % 5 == 0 or iteration == 30:
            print(f"iteration {iteration:02d} phase-time MSE {loss.item():.6f}")

    RESULTS.mkdir(exist_ok=True)
    torch.save(
        {"lon": model.lon, "lat": model.lat, "depth": model.depth, "vp": model.vp.detach(), "vs": model.vs.detach()},
        RESULTS / "model_inverted.pt",
    )
    plot_progress(true, initial, model, loss_history, RESULTS / "inversion_progress.png")
    print(f"saved final model to {RESULTS}; loss {initial_loss:.6f} -> {loss.item():.6f}")


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
            image = axis.imshow(field.T, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
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


if __name__ == "__main__":
    main()
