"""Generate a small station catalog for the synthetic experiment."""

from pathlib import Path

import pandas as pd

DATA = Path("data")
DATA.mkdir(exist_ok=True)

stations = pd.DataFrame(
    [
        ("STA01", -120.28, 34.78, 0.0),
        ("STA02", -119.72, 34.82, 0.0),
        ("STA03", -120.22, 35.27, 0.0),
        ("STA04", -119.76, 35.23, 0.0),
    ],
    columns=["station_id", "longitude", "latitude", "depth_km"],
)
stations.to_csv(DATA / "stations.csv", index=False)
print(f"saved {len(stations)} stations to {DATA}")
