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
DEFAULT_NUM_EVENTS = 500
DEFAULT_SEED = 2
DEFAULT_LON_MIN = -120.6
DEFAULT_LON_MAX = -118.0
DEFAULT_LAT_MIN = 33.8
DEFAULT_LAT_MAX = 36.1
DEFAULT_DEPTH_MIN = 2.0
DEFAULT_DEPTH_MAX = 30.0


def require_inputs():
    model_path, stations_path = DATA / "model_initial.pt", DATA / "stations.csv"
    missing = [path for path in (model_path, stations_path) if not path.is_file()]
    if missing:
        paths = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"missing geometry inputs:\n  {paths}\nrun python examples/00_gen_velocity.py and 01_gen_stations.py first"
        )
    return torch.load(model_path, weights_only=True), pd.read_csv(stations_path, dtype={"station_id": str})


def validate_acquisition_region(model, lon_min, lon_max, lat_min, lat_max, depth_min, depth_max):
    lon, lat, depth = model["lon"], model["lat"], model["depth"]
    if not (lon[0].item() < lon_min < lon_max < lon[-1].item()):
        raise ValueError(
            f"longitude region [{lon_min}, {lon_max}] must be strictly inside "
            f"the model [{lon[0].item()}, {lon[-1].item()}]"
        )
    if not (lat[0].item() < lat_min < lat_max < lat[-1].item()):
        raise ValueError(
            f"latitude region [{lat_min}, {lat_max}] must be strictly inside "
            f"the model [{lat[0].item()}, {lat[-1].item()}]"
        )
    if not (depth[0].item() < depth_min < depth_max < depth[-1].item()):
        raise ValueError(
            f"depth region [{depth_min}, {depth_max}] must be strictly inside "
            f"the model [{depth[0].item()}, {depth[-1].item()}]"
        )


def generate_events(model, num_events, seed, lon_min, lon_max, lat_min, lat_max, depth_min, depth_max):
    if num_events < 1:
        raise ValueError("num-events must be positive")
    validate_acquisition_region(model, lon_min, lon_max, lat_min, lat_max, depth_min, depth_max)
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


def plot_geometry(model, stations, events, lon_min, lon_max, lat_min, lat_max, path):
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axis.plot(
        [model["lon"][0], model["lon"][-1], model["lon"][-1], model["lon"][0], model["lon"][0]],
        [model["lat"][0], model["lat"][0], model["lat"][-1], model["lat"][-1], model["lat"][0]],
        color="black", linewidth=1.2, label="model boundary",
    )
    axis.plot(
        [lon_min, lon_max, lon_max, lon_min, lon_min],
        [lat_min, lat_min, lat_max, lat_max, lat_min],
        color="tab:gray", linestyle="--", linewidth=1.0, label="acquisition region",
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
    parser.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX)
    parser.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX)
    parser.add_argument("--depth-min", type=float, default=DEFAULT_DEPTH_MIN)
    parser.add_argument("--depth-max", type=float, default=DEFAULT_DEPTH_MAX)
    return parser.parse_args()


def main():
    args = parse_arguments()
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    model, stations = require_inputs()
    events = generate_events(
        model,
        args.num_events,
        args.seed,
        args.lon_min,
        args.lon_max,
        args.lat_min,
        args.lat_max,
        args.depth_min,
        args.depth_max,
    )
    path = DATA / "events.csv"
    events.to_csv(path, index=False)
    plot_geometry(model, stations, events, args.lon_min, args.lon_max, args.lat_min, args.lat_max, FIGURES / "geometry.png")
    print(f"saved {len(events)} events with seed {args.seed} to {path}")
    print_ranges(events)
    print(f"saved acquisition geometry to {FIGURES / 'geometry.png'}")


if __name__ == "__main__":
    main()
