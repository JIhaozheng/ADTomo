"""Generate reproducible surface stations inside a chosen global region."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-stations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lon-min", type=float, default=-120.6)
    parser.add_argument("--lon-max", type=float, default=-118.0)
    parser.add_argument("--lat-min", type=float, default=33.8)
    parser.add_argument("--lat-max", type=float, default=36.1)
    args = parser.parse_args()
    if args.num_stations < 1:
        raise ValueError("num-stations must be positive")

    model_path = DATA / "model_initial.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"missing {model_path}; run 00_gen_velocity.py first")
    model = torch.load(model_path, weights_only=True)
    if not (model["lon"][0] < args.lon_min < args.lon_max < model["lon"][-1]):
        raise ValueError("station longitude region must lie strictly inside the velocity model")
    if not (model["lat"][0] < args.lat_min < args.lat_max < model["lat"][-1]):
        raise ValueError("station latitude region must lie strictly inside the velocity model")

    rng = np.random.default_rng(args.seed)
    stations = pd.DataFrame({
        "station_id": [f"STA{i:03d}" for i in range(args.num_stations)],
        "longitude": rng.uniform(args.lon_min, args.lon_max, args.num_stations),
        "latitude": rng.uniform(args.lat_min, args.lat_max, args.num_stations),
        "depth_km": np.zeros(args.num_stations),
    })
    DATA.mkdir(exist_ok=True)
    stations.to_csv(DATA / "stations.csv", index=False)
    print(f"saved {len(stations)} stations to {DATA / 'stations.csv'}")
    print(f"longitude=[{stations.longitude.min():.4f}, {stations.longitude.max():.4f}], latitude=[{stations.latitude.min():.4f}, {stations.latitude.max():.4f}]")


if __name__ == "__main__":
    main()
