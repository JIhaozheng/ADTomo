"""Generate background and smooth checkerboard Vp/Vs models for the example."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

CHECKER_AMPLITUDE = 0.05
WAVELENGTH_LON = 1.5
WAVELENGTH_LAT = 1.5
WAVELENGTH_DEPTH = 20.0
DIAGNOSTIC_DEPTH_KM = 15.0


def smooth_checkerboard(lon, lat, depth, wavelength_lon, wavelength_lat, wavelength_depth):
    """Return a continuous cosine checkerboard in physical model coordinates."""
    depth_grid, lat_grid, lon_grid = torch.meshgrid(depth, lat, lon, indexing="ij")
    lon_phase = 2.0 * torch.pi * (lon_grid - lon[0]) / wavelength_lon
    lat_phase = 2.0 * torch.pi * (lat_grid - lat[0]) / wavelength_lat
    depth_phase = 2.0 * torch.pi * (depth_grid - depth[0]) / wavelength_depth
    return torch.cos(lon_phase) * torch.cos(lat_phase) * torch.cos(depth_phase)


def build_models(
    amplitude=CHECKER_AMPLITUDE,
    wavelength_lon=WAVELENGTH_LON,
    wavelength_lat=WAVELENGTH_LAT,
    wavelength_depth=WAVELENGTH_DEPTH,
):
    """Construct an unperturbed background and its smooth checkerboard truth."""

    lon = torch.arange(-121.0, -117.5, 0.1, dtype=torch.float64)
    lat = torch.arange(33.5, 36.5, 0.1, dtype=torch.float64)
    depth = torch.arange(-15.0, 50.1, 5.0, dtype=torch.float64)
    depth_grid, _, _ = torch.meshgrid(depth, lat, lon, indexing="ij")
    vp_background = 5.5 + 0.03 * depth_grid.clamp_min(0.0)
    vs_background = vp_background / 1.73
    checker = smooth_checkerboard(lon, lat, depth, wavelength_lon, wavelength_lat, wavelength_depth)

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
    lat_index = int(relative.abs().amax(dim=(0, 2)).argmax())
    lon_index = int(relative.abs().amax(dim=(0, 1)).argmax())
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
    parser.add_argument("--wavelength-lon", type=float, default=WAVELENGTH_LON, help="longitude wavelength in degrees")
    parser.add_argument("--wavelength-lat", type=float, default=WAVELENGTH_LAT, help="latitude wavelength in degrees")
    parser.add_argument("--wavelength-depth", type=float, default=WAVELENGTH_DEPTH, help="depth wavelength in km")
    return parser.parse_args()


def main():
    args = parse_arguments()
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    model_initial, model_true, checker = build_models(
        args.amplitude, args.wavelength_lon, args.wavelength_lat, args.wavelength_depth
    )
    validate_models(model_initial, model_true, checker, args.amplitude)
    torch.save(model_initial, DATA / "model_initial.pt")
    torch.save(model_true, DATA / "model_true.pt")
    plot_checkerboard(model_initial, model_true, args.amplitude, FIGURES / "checkerboard.png")
    print(
        f"saved initial and smooth {args.amplitude:.0%} checkerboard models with "
        f"wavelengths {args.wavelength_lon} deg / {args.wavelength_lat} deg / "
        f"{args.wavelength_depth} km (lon/lat/depth) to {DATA}"
    )
    print(f"checker range: [{checker.min().item():.6f}, {checker.max().item():.6f}]")
    print(f"saved checkerboard diagnostic to {FIGURES / 'checkerboard.png'}")


if __name__ == "__main__":
    main()
