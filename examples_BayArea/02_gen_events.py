"""Generate reproducible events and plot the synthetic acquisition geometry."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


def plot_geometry(model, stations, events, args):
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axis.plot([model["lon"][0], model["lon"][-1], model["lon"][-1], model["lon"][0], model["lon"][0]], [model["lat"][0], model["lat"][0], model["lat"][-1], model["lat"][-1], model["lat"][0]], "k-", label="model boundary")
    axis.plot([args.lon_min, args.lon_max, args.lon_max, args.lon_min, args.lon_min], [args.lat_min, args.lat_min, args.lat_max, args.lat_max, args.lat_min], "--", color="gray", label="acquisition region")
    scatter = axis.scatter(events.longitude, events.latitude, c=events.depth_km, cmap="viridis", s=22, label="events")
    axis.scatter(stations.longitude, stations.latitude, marker="^", color="tab:red", edgecolor="black", s=55, label="surface stations")
    axis.set(xlabel="longitude (deg)", ylabel="latitude (deg)", title="Synthetic acquisition geometry")
    axis.legend(loc="best")
    figure.colorbar(scatter, ax=axis, label="event depth (km)")
    figure.savefig(FIGURES / "geometry.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-events", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--lon-min", type=float, default=-120.6)
    parser.add_argument("--lon-max", type=float, default=-118.0)
    parser.add_argument("--lat-min", type=float, default=33.8)
    parser.add_argument("--lat-max", type=float, default=36.1)
    parser.add_argument("--depth-min", type=float, default=2.0)
    parser.add_argument("--depth-max", type=float, default=30.0)
    args = parser.parse_args()
    if args.num_events < 1:
        raise ValueError("num-events must be positive")

    model_path, stations_path = DATA / "model_initial.pt", DATA / "stations.csv"
    if not model_path.is_file() or not stations_path.is_file():
        raise FileNotFoundError("run 00_gen_velocity.py and 01_gen_stations.py first")
    model = torch.load(model_path, weights_only=True)
    stations = pd.read_csv(stations_path, dtype={"station_id": str})
    if not (model["lon"][0] < args.lon_min < args.lon_max < model["lon"][-1]):
        raise ValueError("event longitude region must lie strictly inside the velocity model")
    if not (model["lat"][0] < args.lat_min < args.lat_max < model["lat"][-1]):
        raise ValueError("event latitude region must lie strictly inside the velocity model")
    if not (model["depth"][0] < args.depth_min < args.depth_max < model["depth"][-1]):
        raise ValueError("event depth region must lie strictly inside the velocity model")

    rng = np.random.default_rng(args.seed)
    origin = pd.Timestamp("2026-09-13T12:00:00.000")
    events = pd.DataFrame({
        "event_id": [f"EV{i:03d}" for i in range(args.num_events)],
        "event_time": [(origin + pd.Timedelta(seconds=30 * i)).isoformat(timespec="milliseconds") for i in range(args.num_events)],
        "longitude": rng.uniform(args.lon_min, args.lon_max, args.num_events),
        "latitude": rng.uniform(args.lat_min, args.lat_max, args.num_events),
        "depth_km": rng.uniform(args.depth_min, args.depth_max, args.num_events),
    })
    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    events.to_csv(DATA / "events.csv", index=False)
    plot_geometry(model, stations, events, args)
    print(f"saved {len(events)} events to {DATA / 'events.csv'}")
    print(f"longitude=[{events.longitude.min():.4f}, {events.longitude.max():.4f}], latitude=[{events.latitude.min():.4f}, {events.latitude.max():.4f}], depth=[{events.depth_km.min():.2f}, {events.depth_km.max():.2f}] km")


if __name__ == "__main__":
    main()
