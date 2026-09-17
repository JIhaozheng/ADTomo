"""Generate background and smooth checkerboard Vp/Vs models for the example."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

CHECKER_AMPLITUDE = 0.05
CHECKER_PERIODS_LON = 3
CHECKER_PERIODS_LAT = 3
CHECKER_PERIODS_DEPTH = 2
DIAGNOSTIC_DEPTH_KM = 15.0


def smooth_checkerboard(lon, lat, depth, periods_lon, periods_lat, periods_depth):
    """Return a continuous cosine checkerboard in physical model coordinates."""
    depth_grid, lat_grid, lon_grid = torch.meshgrid(depth, lat, lon, indexing="ij")
    lon_phase = 2.0 * torch.pi * periods_lon * (lon_grid - lon[0]) / (lon[-1] - lon[0])
    lat_phase = 2.0 * torch.pi * periods_lat * (lat_grid - lat[0]) / (lat[-1] - lat[0])
    depth_phase = 2.0 * torch.pi * periods_depth * (depth_grid - depth[0]) / (depth[-1] - depth[0])
    return torch.cos(lon_phase) * torch.cos(lat_phase) * torch.cos(depth_phase)


def build_models(
    amplitude=CHECKER_AMPLITUDE,
    periods_lon=CHECKER_PERIODS_LON,
    periods_lat=CHECKER_PERIODS_LAT,
    periods_depth=CHECKER_PERIODS_DEPTH,
):
    """Construct an unperturbed background and its smooth checkerboard truth."""
    if not 0.0 < amplitude < 1.0:
        raise ValueError("amplitude must be in (0, 1)")
    if min(periods_lon, periods_lat, periods_depth) < 1:
        raise ValueError("all checkerboard period counts must be positive")

    lon = torch.arange(-120.8, -119.19, 0.1, dtype=torch.float64)
    lat = torch.arange(34.2, 35.81, 0.1, dtype=torch.float64)
    depth = torch.arange(-15.0, 50.1, 5.0, dtype=torch.float64)
    depth_grid, _, _ = torch.meshgrid(depth, lat, lon, indexing="ij")
    vp_background = 5.5 + 0.03 * depth_grid.clamp_min(0.0)
    vs_background = vp_background / 1.73
    checker = smooth_checkerboard(lon, lat, depth, periods_lon, periods_lat, periods_depth)

    model_initial = {"lon": lon, "lat": lat, "depth": depth, "vp": vp_background, "vs": vs_background}
    model_true = {
        "lon": lon.clone(),
        "lat": lat.clone(),
        "depth": depth.clone(),
        "vp": vp_background * (1.0 + amplitude * checker),
        "vs": vs_background * (1.0 + amplitude * checker),
    }
    return model_initial, model_true, checker


def validate_models(model_initial, model_true, checker, amplitude):
    if not (checker.min() < 0.0 and checker.max() > 0.0 and checker.abs().max() <= 1.0 + 1e-12):
        raise AssertionError("checkerboard must have bounded positive and negative lobes")
    for phase in ("vp", "vs"):
        relative = (model_true[phase] - model_initial[phase]) / model_initial[phase]
        if not torch.allclose(relative, amplitude * checker):
            raise AssertionError(f"{phase} relative perturbation does not match the checkerboard")
        intermediate = (relative.abs() > amplitude * 0.1) & (relative.abs() < amplitude * 0.9)
        if not intermediate.any():
            raise AssertionError("checkerboard contains no intermediate values; it is not smooth")
        if model_true[phase].min() <= 0.0:
            raise AssertionError(f"{phase} must remain positive")


def plot_checkerboard(model_initial, model_true, amplitude, path):
    relative = (model_true["vp"] - model_initial["vp"]) / model_initial["vp"]
    depth_index = int(torch.argmin((model_initial["depth"] - DIAGNOSTIC_DEPTH_KM).abs()))
    lat_index = len(model_initial["lat"]) // 2
    lon_index = len(model_initial["lon"]) // 2
    lon_extent = [model_initial["lon"][0].item(), model_initial["lon"][-1].item()]
    lat_extent = [model_initial["lat"][0].item(), model_initial["lat"][-1].item()]
    depth_extent = [model_initial["depth"][-1].item(), model_initial["depth"][0].item()]
    depth_km = model_initial["depth"][depth_index].item()
    slices = (
        (relative[depth_index], [*lon_extent, *lat_extent], "longitude (deg)", "latitude (deg)", f"horizontal at {depth_km:.1f} km"),
        (relative[:, lat_index, :], [*lon_extent, *depth_extent], "longitude (deg)", "depth (km)", "longitude-depth"),
        (relative[:, :, lon_index], [*lat_extent, *depth_extent], "latitude (deg)", "depth (km)", "latitude-depth"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for index, (axis, (field, extent, xlabel, ylabel, title)) in enumerate(zip(axes, slices)):
        image = axis.imshow(
            field,
            origin="upper",
            extent=extent,
            aspect="equal" if index == 0 else "auto",
            interpolation="bilinear",
            cmap="seismic",
            vmin=-amplitude,
            vmax=amplitude,
        )
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        figure.colorbar(image, ax=axis, label="Vp relative perturbation")
    figure.suptitle("Smooth checkerboard truth")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amplitude", type=float, default=CHECKER_AMPLITUDE)
    parser.add_argument("--periods-lon", type=int, default=CHECKER_PERIODS_LON)
    parser.add_argument("--periods-lat", type=int, default=CHECKER_PERIODS_LAT)
    parser.add_argument("--periods-depth", type=int, default=CHECKER_PERIODS_DEPTH)
    return parser.parse_args()


def main():
    args = parse_arguments()
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    model_initial, model_true, checker = build_models(
        args.amplitude, args.periods_lon, args.periods_lat, args.periods_depth
    )
    validate_models(model_initial, model_true, checker, args.amplitude)
    torch.save(model_initial, DATA / "model_initial.pt")
    torch.save(model_true, DATA / "model_true.pt")
    plot_checkerboard(model_initial, model_true, args.amplitude, FIGURES / "checkerboard.png")
    print(
        f"saved initial and smooth {args.amplitude:.0%} checkerboard models with "
        f"{args.periods_lon}/{args.periods_lat}/{args.periods_depth} lon/lat/depth periods to {DATA}"
    )
    print(f"checker range: [{checker.min().item():.6f}, {checker.max().item():.6f}]")
    print(f"saved checkerboard diagnostic to {FIGURES / 'checkerboard.png'}")


if __name__ == "__main__":
    main()
