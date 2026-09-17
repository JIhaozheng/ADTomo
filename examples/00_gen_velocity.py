"""Generate background and checkerboard Vp/Vs models for the example."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

CHECKER_AMPLITUDE = 0.05
CHECKER_LON_NODES = 4
CHECKER_LAT_NODES = 4
CHECKER_DEPTH_NODES = 2
DIAGNOSTIC_DEPTH_KM = 15.0


def make_checkerboard(shape, dtype):
    """Return alternating -1/+1 blocks in (depth, latitude, longitude) order."""
    depth_index, lat_index, lon_index = torch.meshgrid(
        torch.arange(shape[0]),
        torch.arange(shape[1]),
        torch.arange(shape[2]),
        indexing="ij",
    )
    parity = (
        depth_index // CHECKER_DEPTH_NODES
        + lat_index // CHECKER_LAT_NODES
        + lon_index // CHECKER_LON_NODES
    ) % 2
    return (2 * parity - 1).to(dtype=dtype)


def build_models():
    """Construct an unperturbed background and its checkerboard truth."""
    lon = torch.arange(-120.8, -119.19, 0.1, dtype=torch.float64)
    lat = torch.arange(34.2, 35.81, 0.1, dtype=torch.float64)
    depth = torch.arange(-15.0, 50.1, 5.0, dtype=torch.float64)
    depth_grid, lat_grid, lon_grid = torch.meshgrid(depth, lat, lon, indexing="ij")

    # The public example deliberately uses a simple, self-contained 1-D background.
    vp_background = 5.5 + 0.03 * depth_grid.clamp_min(0.0)
    vs_background = vp_background / 1.73
    checker = make_checkerboard(vp_background.shape, vp_background.dtype)

    model_initial = {
        "lon": lon,
        "lat": lat,
        "depth": depth,
        "vp": vp_background,
        "vs": vs_background,
    }
    model_true = {
        "lon": lon.clone(),
        "lat": lat.clone(),
        "depth": depth.clone(),
        "vp": vp_background * (1.0 + CHECKER_AMPLITUDE * checker),
        "vs": vs_background * (1.0 + CHECKER_AMPLITUDE * checker),
    }
    return model_initial, model_true


def plot_checkerboard(model_initial, model_true, path):
    depth_index = int(torch.argmin((model_initial["depth"] - DIAGNOSTIC_DEPTH_KM).abs()))
    depth_km = model_initial["depth"][depth_index].item()
    extent = [
        model_initial["lon"][0].item(),
        model_initial["lon"][-1].item(),
        model_initial["lat"][0].item(),
        model_initial["lat"][-1].item(),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, name in zip(axes, ("vp", "vs")):
        relative = (
            model_true[name][depth_index] - model_initial[name][depth_index]
        ) / model_initial[name][depth_index]
        image = axis.imshow(
            relative,
            origin="lower",
            extent=extent,
            cmap="seismic",
            vmin=-CHECKER_AMPLITUDE,
            vmax=CHECKER_AMPLITUDE,
        )
        axis.set(
            title=f"{name.upper()} relative perturbation",
            xlabel="longitude (deg)",
            ylabel="latitude (deg)",
        )
        figure.colorbar(image, ax=axis, label="relative velocity")
    figure.suptitle(f"Checkerboard truth at {depth_km:.1f} km depth")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    model_initial, model_true = build_models()
    torch.save(model_initial, DATA / "model_initial.pt")
    torch.save(model_true, DATA / "model_true.pt")
    plot_checkerboard(model_initial, model_true, FIGURES / "checkerboard.png")
    print(
        f"saved initial and {CHECKER_AMPLITUDE:.0%} checkerboard models "
        f"with shape {tuple(model_initial['vp'].shape)} to {DATA}"
    )
    print(f"saved checkerboard diagnostic to {FIGURES / 'checkerboard.png'}")


if __name__ == "__main__":
    main()
