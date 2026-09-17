"""Generate reproducible hypocenters and origin times for the example."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RANDOM_SEED = 7
EVENT_COUNT = 12


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    first_origin = pd.Timestamp("2026-09-13T12:00:00.000")
    events = pd.DataFrame(
        {
            "event_id": [f"EV{i:03d}" for i in range(EVENT_COUNT)],
            "event_time": [
                (first_origin + pd.Timedelta(seconds=30 * i)).isoformat(timespec="milliseconds")
                for i in range(EVENT_COUNT)
            ],
            "longitude": rng.uniform(-120.28, -119.72, EVENT_COUNT),
            "latitude": rng.uniform(34.72, 35.28, EVENT_COUNT),
            "depth_km": rng.uniform(4.0, 22.0, EVENT_COUNT),
        }
    )
    path = DATA / "events.csv"
    events.to_csv(path, index=False)
    print(f"saved {len(events)} events with seed {RANDOM_SEED} to {path}")


if __name__ == "__main__":
    main()
