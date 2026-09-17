"""Create the initial spherical Vp/Vs model for the supplied real catalog."""

from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_HORIZONTAL_SPACING_DEG = 0.05
MODEL_DEPTH_SPACING_KM = 2.0

# Fixed geographic model region for the complete supplied catalog.
LON_MIN_DEG, LON_MAX_DEG = -124.2, -118.6
LAT_MIN_DEG, LAT_MAX_DEG = 34.5, 40.0
DEPTH_MIN_KM, DEPTH_MAX_KM = -6.0, 45.0


lon = torch.arange(
    LON_MIN_DEG, LON_MAX_DEG + 0.5 * MODEL_HORIZONTAL_SPACING_DEG, MODEL_HORIZONTAL_SPACING_DEG, dtype=torch.float64
)
lat = torch.arange(
    LAT_MIN_DEG, LAT_MAX_DEG + 0.5 * MODEL_HORIZONTAL_SPACING_DEG, MODEL_HORIZONTAL_SPACING_DEG, dtype=torch.float64
)
depth = torch.arange(
    DEPTH_MIN_KM, DEPTH_MAX_KM + 0.5 * MODEL_DEPTH_SPACING_KM, MODEL_DEPTH_SPACING_KM, dtype=torch.float64
)
depth_grid, _, _ = torch.meshgrid(depth, lat, lon, indexing="ij")
vp = 5.5 + 0.03 * depth_grid.clamp_min(0.0)
torch.save({"lon": lon, "lat": lat, "depth": depth, "vp": vp, "vs": vp / 1.73}, DATA / "model_initial.pt")
print(f"saved initial model with shape {tuple(vp.shape)} to {DATA}")
