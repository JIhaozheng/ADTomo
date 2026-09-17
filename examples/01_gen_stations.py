"""Generate a reproducible, broad surface-station catalog for the example."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEFAULT_NUM_STATIONS = 12
DEFAULT_SEED = 1
HORIZONTAL_MARGIN = 0.05


def require_model():
    path = DATA / "model_initial.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run python examples/00_gen_velocity.py first")
    return torch.load(path, weights_only=True)


def horizontal_bounds(model, margin=HORIZONTAL_MARGIN):
    lon, lat = model["lon"], model["lat"]
    lon_pad = margin * (lon[-1] - lon[0]).item()
    lat_pad = margin * (lat[-1] - lat[0]).item()
    return lon[0].item() + lon_pad, lon[-1].item() - lon_pad, lat[0].item() + lat_pad, lat[-1].item() - lat_pad


def generate_stations(model, num_stations, seed):
    if num_stations < 1:
        raise ValueError("num-stations must be positive")
    lon_min, lon_max, lat_min, lat_max = horizontal_bounds(model)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "station_id": [f"STA{i:03d}" for i in range(num_stations)],
            "longitude": rng.uniform(lon_min, lon_max, num_stations),
            "latitude": rng.uniform(lat_min, lat_max, num_stations),
            "depth_km": np.zeros(num_stations),
        }
    )


def print_ranges(stations):
    print(f"station longitude range: [{stations.longitude.min():.4f}, {stations.longitude.max():.4f}]")
    print(f"station latitude range:  [{stations.latitude.min():.4f}, {stations.latitude.max():.4f}]")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-stations", type=int, default=DEFAULT_NUM_STATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = parse_arguments()
    DATA.mkdir(parents=True, exist_ok=True)
    stations = generate_stations(require_model(), args.num_stations, args.seed)
    path = DATA / "stations.csv"
    stations.to_csv(path, index=False)
    print(f"saved {len(stations)} surface stations with seed {args.seed} to {path}")
    print_ranges(stations)


if __name__ == "__main__":
    main()
