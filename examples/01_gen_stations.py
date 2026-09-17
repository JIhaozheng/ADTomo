"""Generate a reproducible, broad surface-station catalog for the example."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEFAULT_NUM_STATIONS = 50
DEFAULT_SEED = 1
DEFAULT_LON_MIN = -120.6
DEFAULT_LON_MAX = -118.0
DEFAULT_LAT_MIN = 33.8
DEFAULT_LAT_MAX = 36.1


def require_model():
    path = DATA / "model_initial.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run python examples/00_gen_velocity.py first")
    return torch.load(path, weights_only=True)


def validate_horizontal_region(model, lon_min, lon_max, lat_min, lat_max):
    lon, lat = model["lon"], model["lat"]
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


def generate_stations(model, num_stations, seed, lon_min, lon_max, lat_min, lat_max):
    if num_stations < 1:
        raise ValueError("num-stations must be positive")
    validate_horizontal_region(model, lon_min, lon_max, lat_min, lat_max)
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
    parser.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX)
    parser.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX)
    return parser.parse_args()


def main():
    args = parse_arguments()
    DATA.mkdir(parents=True, exist_ok=True)
    stations = generate_stations(
        require_model(), args.num_stations, args.seed, args.lon_min, args.lon_max, args.lat_min, args.lat_max
    )
    path = DATA / "stations.csv"
    stations.to_csv(path, index=False)
    print(f"saved {len(stations)} surface stations with seed {args.seed} to {path}")
    print_ranges(stations)


if __name__ == "__main__":
    main()
