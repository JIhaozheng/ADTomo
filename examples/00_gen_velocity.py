"""Generate true and perturbed initial spherical velocity models."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = Path(__file__).resolve().parent / "data"


def save_model(path, lon, lat, depth, vp, vs):
    torch.save({"lon": lon, "lat": lat, "depth": depth, "vp": vp, "vs": vs}, path)


def main():
    DATA.mkdir(exist_ok=True)
    lon = torch.arange(-120.8, -119.19, 0.1, dtype=torch.float64)
    lat = torch.arange(34.2, 35.81, 0.1, dtype=torch.float64)
    depth = torch.arange(-15.0, 50.1, 5.0, dtype=torch.float64)
    z, y, x = torch.meshgrid(depth, lat, lon, indexing="ij")
    vp0 = 5.5 + 0.03 * z.clamp_min(0.0)
    anomaly = 0.35 * torch.exp(-((x + 120.0) / 0.22) ** 2 - ((y - 35.0) / 0.20) ** 2 - ((z - 15.0) / 10.0) ** 2)
    vp_true = vp0 + anomaly
    vs_true = vp_true / 1.73
    vp_initial = vp0 + 0.05
    vs_initial = vp_initial / 1.73
    save_model(DATA / "model_true.pt", lon, lat, depth, vp_true, vs_true)
    save_model(DATA / "model_initial.pt", lon, lat, depth, vp_initial, vs_initial)
    print(f"saved models with shape {tuple(vp_true.shape)} to {DATA}")


if __name__ == "__main__":
    main()
