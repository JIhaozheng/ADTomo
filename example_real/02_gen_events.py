"""Generate fixed hypocenters and catalog origin times."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
DATA.mkdir(exist_ok=True)

rng = np.random.default_rng(7)
event_time = pd.Timestamp("2026-09-13T12:00:00.000")
events = pd.DataFrame(
    {
        "event_id": [f"EV{i:03d}" for i in range(12)],
        "event_time": [
            (event_time + pd.Timedelta(seconds=30 * i)).isoformat(timespec="milliseconds") for i in range(12)
        ],
        "longitude": rng.uniform(-120.28, -119.72, 12),
        "latitude": rng.uniform(34.72, 35.28, 12),
        "depth_km": rng.uniform(4.0, 22.0, 12),
    }
)
events.to_csv(DATA / "events.csv", index=False)
print(f"saved {len(events)} events to {DATA}")
