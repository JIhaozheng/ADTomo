"""Generate reproducible, broad hypocenters and an acquisition-geometry figure."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
DEFAULT_NUM_EVENTS = 50
DEFAULT_SEED = 2
HORIZONTAL_MARGIN = 0.10
DEPTH_MARGIN = 0.10
MIN_EVENT_DEPTH_KM = 2.0
FORWARD_GRID_DEPTH_BUFFER_KM = 15.0


def require_inputs():
    model_path, stations_path = DATA / "model_initial.pt", DATA / "stations.csv"
    missing = [path for path in (model_path, stations_path) if not path.is_file()]
    if missing:
        paths = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"missing geometry inputs:\n  {paths}\nrun python examples/00_gen_velocity.py and 01_gen_stations.py first"
        )
    return torch.load(model_path, weights_only=True), pd.read_csv(stations_path, dtype={"station_id": str})


def event_bounds(model):
    lon, lat, depth = model["lon"], model["lat"], model["depth"]
    lon_pad = HORIZONTAL_MARGIN * (lon[-1] - lon[0]).item()
    lat_pad = HORIZONTAL_MARGIN * (lat[-1] - lat[0]).item()
    depth_pad = DEPTH_MARGIN * (depth[-1] - depth[0]).item()
    depth_min = max(MIN_EVENT_DEPTH_KM, depth[0].item() + depth_pad)
    # Reserve room for the station-aligned grid's downward padding and curvature.
    depth_max = depth[-1].item() - max(depth_pad, FORWARD_GRID_DEPTH_BUFFER_KM)
    if depth_min >= depth_max:
        raise ValueError("model depth range is too small for the event safety margin")
    return lon[0].item() + lon_pad, lon[-1].item() - lon_pad, lat[0].item() + lat_pad, lat[-1].item() - lat_pad, depth_min, depth_max


def generate_events(model, num_events, seed):
    if num_events < 1:
        raise ValueError("num-events must be positive")
    lon_min, lon_max, lat_min, lat_max, depth_min, depth_max = event_bounds(model)
    rng = np.random.default_rng(seed)
    first_origin = pd.Timestamp("2026-09-13T12:00:00.000")
    return pd.DataFrame(
        {
            "event_id": [f"EV{i:03d}" for i in range(num_events)],
            "event_time": [(first_origin + pd.Timedelta(seconds=30 * i)).isoformat(timespec="milliseconds") for i in range(num_events)],
            "longitude": rng.uniform(lon_min, lon_max, num_events),
            "latitude": rng.uniform(lat_min, lat_max, num_events),
            "depth_km": rng.uniform(depth_min, depth_max, num_events),
        }
    )


def plot_geometry(model, stations, events, path):
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axis.plot(
        [model["lon"][0], model["lon"][-1], model["lon"][-1], model["lon"][0], model["lon"][0]],
        [model["lat"][0], model["lat"][0], model["lat"][-1], model["lat"][-1], model["lat"][0]],
        color="black", linewidth=1.2, label="model boundary",
    )
    scatter = axis.scatter(events.longitude, events.latitude, c=events.depth_km, cmap="viridis", s=22, alpha=0.8, label="events")
    axis.scatter(stations.longitude, stations.latitude, marker="^", color="tab:red", edgecolor="black", s=55, label="surface stations")
    axis.set(xlabel="longitude (deg)", ylabel="latitude (deg)", title="Synthetic acquisition geometry")
    axis.legend(loc="best")
    figure.colorbar(scatter, ax=axis, label="event depth (km)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def print_ranges(events):
    print(f"event longitude range: [{events.longitude.min():.4f}, {events.longitude.max():.4f}]")
    print(f"event latitude range:  [{events.latitude.min():.4f}, {events.latitude.max():.4f}]")
    print(f"event depth range:     [{events.depth_km.min():.4f}, {events.depth_km.max():.4f}] km")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-events", type=int, default=DEFAULT_NUM_EVENTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = parse_arguments()
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    model, stations = require_inputs()
    events = generate_events(model, args.num_events, args.seed)
    path = DATA / "events.csv"
    events.to_csv(path, index=False)
    plot_geometry(model, stations, events, FIGURES / "geometry.png")
    print(f"saved {len(events)} events with seed {args.seed} to {path}")
    print_ranges(events)
    print(f"saved acquisition geometry to {FIGURES / 'geometry.png'}")


if __name__ == "__main__":
    main()
