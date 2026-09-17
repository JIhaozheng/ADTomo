"""Generate the station catalog for the synthetic example."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    stations = pd.DataFrame(
        [
            ("STA01", -120.28, 34.78, 0.0),
            ("STA02", -119.72, 34.82, 0.0),
            ("STA03", -120.22, 35.27, 0.0),
            ("STA04", -119.76, 35.23, 0.0),
        ],
        columns=["station_id", "longitude", "latitude", "depth_km"],
    )
    path = DATA / "stations.csv"
    stations.to_csv(path, index=False)
    print(f"saved {len(stations)} stations to {path}")


if __name__ == "__main__":
    main()
