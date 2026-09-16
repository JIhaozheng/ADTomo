"""Run the controlled three-way 3-D source-correction experiment.

Only the C++ backward module changes between variants.  The production forward
solver, ForwardGrid sampling, tensor permutations, receiver interpolation, and
Taylor directions are shared exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from adtomo import ForwardGrid, VelocityModel, ecef_to_spherical, local_basis, local_to_ecef, spherical_to_ecef

from .loader import PRODUCTION, VARIANTS, load_extensions, sha256, validate_sources


DTYPE = torch.float64
EPSILONS = tuple(10.0 ** -i for i in range(1, 8))
VARIANT_ORDER = ("full", "no_lu", "bulk_only")
SOURCE_CASES = {
    "aligned": (0.0, 0.0, 0.0),
    "fractional_center": (0.4, 0.5, 0.6),
    "fractional_corner": (0.1, 0.15, 0.2),
}
RECEIVER_XYZ = (14, 14, 14)


def solve_slowness_variant(op, slowness_xyz: torch.Tensor, source_xyz, spacing: float) -> torch.Tensor:
    """Autograd bridge for a selected extension with C++ xyz tensor ordering."""
    source = torch.as_tensor(source_xyz, dtype=slowness_xyz.dtype, device=slowness_xyz.device)

    class Function(torch.autograd.Function):
        @staticmethod
        def forward(ctx, slowness, h, x, y, z):
            travel_time = op.forward(slowness, h, x, y, z)
            ctx.save_for_backward(travel_time, slowness)
            ctx.h, ctx.source = h, (x, y, z)
            return travel_time

        @staticmethod
        def backward(ctx, grad_output):
            travel_time, slowness = ctx.saved_tensors
            return op.backward(grad_output.contiguous(), travel_time, slowness, ctx.h, *ctx.source), None, None, None, None

    return Function.apply(slowness_xyz.contiguous(), float(spacing), *source.tolist())


def solve_variant(op, velocity_zyx: torch.Tensor, source_xyz, spacing: float) -> torch.Tensor:
    """Production solve_eikonal3d with only the extension dependency injected."""
    slowness_xyz = (1.0 / velocity_zyx).permute(2, 1, 0).contiguous()
    return solve_slowness_variant(op, slowness_xyz, source_xyz, spacing).permute(2, 1, 0)


def smooth_velocity_xyz(shape, amplitude=0.05) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, shape[0], dtype=DTYPE)[:, None, None]
    y = torch.linspace(0.0, 1.0, shape[1], dtype=DTYPE)[None, :, None]
    z = torch.linspace(0.0, 1.0, shape[2], dtype=DTYPE)[None, None, :]
    return 6.0 * (1.0 + amplitude * torch.sin(2 * math.pi * x) * torch.cos(math.pi * y) * torch.sin(math.pi * z))


def local_direction(shape, scale=0.01) -> torch.Tensor:
    return torch.linspace(-scale, scale, int(np.prod(shape)), dtype=DTYPE).reshape(shape)


def _metric(epsilons, r0, r1, objective, directional):
    floor = 100.0 * np.finfo(float).eps * max(1.0, abs(objective), abs(directional))
    valid = [value > floor and math.isfinite(value) for value in r1]
    best = []
    current = []
    for idx, ok in enumerate(valid):
        if ok and (not current or r1[idx] < r1[current[-1]]):
            current.append(idx)
        else:
            if len(current) > len(best):
                best = current
            current = [idx] if ok else []
    if len(current) > len(best):
        best = current
    local = [math.log(r1[i + 1] / r1[i]) / math.log(epsilons[i + 1] / epsilons[i]) for i in range(len(r1) - 1) if r1[i] > 0 and r1[i + 1] > 0]
    if len(best) < 3:
        return {"slope": float("nan"), "r2": float("nan"), "pass": False, "floor": floor, "window": best, "local": local}
    x = np.log(np.asarray([epsilons[i] for i in best]))
    y = np.log(np.asarray([r1[i] for i in best]))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    denom = float(np.square(y - y.mean()).sum())
    r2 = 1.0 if denom == 0 else 1.0 - float(np.square(y - pred).sum()) / denom
    return {"slope": float(slope), "r2": r2, "pass": bool(1.8 <= slope <= 2.2 and r2 >= 0.98), "floor": floor, "window": best, "local": local}


def taylor(objective_fn, base: torch.Tensor, direction: torch.Tensor):
    candidate = base.detach().clone().requires_grad_(True)
    value = objective_fn(candidate)
    value.backward()
    derivative = float((candidate.grad * direction).sum())
    base_value = float(value.detach())
    r0, r1 = [], []
    for epsilon in EPSILONS:
        perturbed = float(objective_fn(base.detach() + epsilon * direction).detach())
        change = perturbed - base_value
        r0.append(abs(change))
        r1.append(abs(change - epsilon * derivative))
    return {"objective": base_value, "directional": derivative, "gradient": candidate.grad.detach(), "r0": r0, "r1": r1, "metric": _metric(EPSILONS, r0, r1, base_value, derivative)}


def check_common_forward(ops, f_xyz, source):
    fields = {name: op.forward(f_xyz.contiguous(), 1.0, *source) for name, op in ops.items()}
    baseline = fields["full"]
    if not all(torch.equal(baseline, field) for field in fields.values()):
        raise RuntimeError("variant forward fields differ; aborting comparison")


def isolated_cases(ops):
    shape = (17, 18, 19)  # xyz ordering used directly by C++.
    direction = local_direction(shape)
    rows, curves = [], {}
    for model_name, velocity in (("constant", torch.full(shape, 6.0, dtype=DTYPE)), ("smooth", smooth_velocity_xyz(shape))):
        f = (1.0 / velocity).contiguous()
        for source_name, frac in SOURCE_CASES.items():
            source = (5.0 + frac[0], 6.0 + frac[1], 7.0 + frac[2])
            check_common_forward(ops, f, source)
            results = {}
            for variant, op in ops.items():
                results[variant] = taylor(lambda candidate, op=op: solve_slowness_variant(op, candidate, source, 1.0)[RECEIVER_XYZ], f, direction)
            full_grad = results["full"]["gradient"]
            for variant, result in results.items():
                rel = float(torch.linalg.vector_norm(result["gradient"] - full_grad) / torch.linalg.vector_norm(full_grad))
                curves[("isolated", model_name, source_name, variant)] = result["r1"]
                for index, epsilon in enumerate(EPSILONS):
                    rows.append({"level": "isolated", "model": model_name, "source": source_name, "variant": variant, "epsilon": epsilon, "r0": result["r0"][index], "r1": result["r1"][index], "local_r1_slope": "" if index == len(EPSILONS) - 1 else result["metric"]["local"][index], "representative_slope": result["metric"]["slope"], "fit_r2": result["metric"]["r2"], "pass": result["metric"]["pass"], "relative_gradient_difference": rel})
    return rows, curves


def _global_axes():
    return (torch.arange(-120.8, -119.19, 0.1, dtype=DTYPE), torch.arange(34.2, 35.81, 0.1, dtype=DTYPE), torch.arange(-15.0, 50.1, 5.0, dtype=DTYPE))


def _global_velocity(shape, smooth):
    if not smooth:
        return torch.full(shape, 6.0, dtype=DTYPE)
    z = torch.linspace(0.0, 1.0, shape[0], dtype=DTYPE)[:, None, None]
    y = torch.linspace(0.0, 1.0, shape[1], dtype=DTYPE)[None, :, None]
    x = torch.linspace(0.0, 1.0, shape[2], dtype=DTYPE)[None, None, :]
    return 6.0 * (1.0 + 0.05 * torch.sin(math.pi * z) * torch.cos(2 * math.pi * y) * torch.sin(2 * math.pi * x))


def _event_lonlatdepth(station, local_xyz):
    station_ecef = spherical_to_ecef(*station)
    basis = local_basis(station[0], station[1])
    ecef = local_to_ecef(torch.as_tensor(local_xyz, dtype=DTYPE), station_ecef, basis)
    lon, lat, depth = ecef_to_spherical(ecef)
    return torch.stack([lon, lat, depth], dim=-1)


def full_geometry(fractions):
    station = torch.tensor([-120.0, 35.0, 10.0], dtype=DTYPE)
    spacing = 5.0
    # The first event controls the negative lower bound and is excluded from MSE.
    anchor = tuple(-spacing * value for value in fractions)
    receivers = ((20.0, 15.0, 10.0), (25.0, 20.0, 15.0), (15.0, 25.0, 25.0))
    events = torch.cat([_event_lonlatdepth(station, anchor).reshape(1, 3), _event_lonlatdepth(station, receivers)], dim=0)
    lon, lat, depth = _global_axes()
    template = VelocityModel(lon, lat, depth, torch.full((len(depth), len(lat), len(lon)), 6.0, dtype=DTYPE), torch.full((len(depth), len(lat), len(lon)), 3.5, dtype=DTYPE), trainable=False)
    grid = ForwardGrid(station, events, template, spacing=spacing)
    expected = torch.tensor([2.0 + value for value in fractions], dtype=DTYPE)
    if not torch.allclose(grid.station_index, expected, rtol=0.0, atol=1e-7):
        raise RuntimeError(f"ForwardGrid source {grid.station_index.tolist()} != requested {expected.tolist()}")
    return lon, lat, depth, grid, torch.tensor([1, 2, 3], dtype=torch.long)


def predict_variant(op, model, grid, event_indices):
    velocity = grid.sample(model.vp)
    travel_time = solve_variant(op, velocity, grid.station_index, grid.spacing)
    return grid.sample_events(travel_time, event_indices=event_indices), travel_time


def full_cases(ops, source_cases=SOURCE_CASES, model_cases=("constant", "smooth"), label="initial"):
    rows, curves = [], {}
    for source_name, fractions in source_cases.items():
        lon, lat, depth, grid, event_indices = full_geometry(fractions)
        for model_name in model_cases:
            vp0 = _global_velocity((len(depth), len(lat), len(lon)), smooth=model_name == "smooth")
            vs0 = vp0 / 1.73
            reference_model = VelocityModel(lon, lat, depth, vp0, vs0, trainable=False)
            common = {variant: predict_variant(op, reference_model, grid, event_indices) for variant, op in ops.items()}
            tt_full = common["full"][1]
            if not all(torch.equal(tt_full, item[1]) and torch.equal(common["full"][0], item[0]) for item in common.values()):
                raise RuntimeError("full-chain forward/interpolation differs between variants")
            observed = common["full"][0].detach() + torch.tensor([0.20, -0.15, 0.10], dtype=DTYPE)
            direction = torch.linspace(-0.1, 0.1, vp0.numel(), dtype=DTYPE).reshape_as(vp0)
            for variant, op in ops.items():
                def objective(candidate, op=op):
                    # VelocityModel clones constructor inputs into Parameters.  A tiny
                    # proxy keeps ``candidate`` as the global model.vp leaf while
                    # preserving the production ForwardGrid.sample(model.vp) chain.
                    model = SimpleNamespace(vp=candidate, vs=vs0)
                    predicted, _ = predict_variant(op, model, grid, event_indices)
                    return (predicted - observed).square().mean()

                result = taylor(objective, vp0, direction)
                curves[("full_chain", model_name, source_name, variant)] = result["r1"]
                for index, epsilon in enumerate(EPSILONS):
                    rows.append({"level": "full_chain", "suite": label, "model": model_name, "source": source_name, "variant": variant, "epsilon": epsilon, "r0": result["r0"][index], "r1": result["r1"][index], "local_r1_slope": "" if index == len(EPSILONS) - 1 else result["metric"]["local"][index], "representative_slope": result["metric"]["slope"], "fit_r2": result["metric"]["r2"], "pass": result["metric"]["pass"], "relative_gradient_difference": ""})
    return rows, curves


def benchmark(ops):
    shape, source, receiver = (41, 31, 21), (20.4, 15.5, 10.6), (35, 25, 17)
    f = (1.0 / smooth_velocity_xyz(shape)).contiguous()
    grad_u = torch.zeros(shape, dtype=DTYPE)
    grad_u[receiver] = 1.0
    fields = {name: op.forward(f, 1.0, *source) for name, op in ops.items()}
    if not all(torch.equal(fields["full"], value) for value in fields.values()):
        raise RuntimeError("benchmark forward fields differ")
    for op in ops.values():
        for _ in range(3):
            op.backward(grad_u, fields["full"], f, 1.0, *source)
    samples = {name: [] for name in VARIANT_ORDER}
    for _ in range(10):
        for name in VARIANT_ORDER:
            start = time.perf_counter_ns()
            ops[name].backward(grad_u, fields[name], f, 1.0, *source)
            samples[name].append((time.perf_counter_ns() - start) / 1e6)
    full_median = float(np.median(samples["full"]))
    return [{"variant": name, "median_ms": float(np.median(values)), "iqr_ms": float(np.percentile(values, 75) - np.percentile(values, 25)), "speed_ratio_to_full": float(np.median(values)) / full_median} for name, values in samples.items()]


def plot_curves(curves, output):
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    groups = {}
    for (level, model, source, variant), values in curves.items():
        groups.setdefault((level, model, source), {})[variant] = values
    for (level, model, source), values in groups.items():
        figure, axis = plt.subplots(figsize=(5.5, 4.0))
        for variant in VARIANT_ORDER:
            axis.loglog(EPSILONS, values[variant], "o-", label=variant)
        reference = values["full"][0] * (np.asarray(EPSILONS) / EPSILONS[0]) ** 2
        axis.loglog(EPSILONS, reference, "k--", label=r"$O(\epsilon^2)$")
        axis.invert_xaxis()
        axis.set(xlabel=r"$\epsilon$", ylabel=r"$R_1$", title=f"{level}: {model}, {source}")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plots / f"{level}_{model}_{source}.png", dpi=180)
        plt.close(figure)


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def recommendation(rows):
    summary = {}
    for row in rows:
        key = (row["level"], row["model"], row["source"], row["variant"])
        summary[key] = bool(row["pass"])
    full_fail = [key for key, passed in summary.items() if key[-1] == "full" and not passed]
    no_lu_fail = [key for key, passed in summary.items() if key[-1] == "no_lu" and not passed and summary.get((*key[:-1], "full"), False)]
    bulk_fractional_fail = [key for key, passed in summary.items() if key[-1] == "bulk_only" and key[2] != "aligned" and not passed and summary.get((*key[:-1], "full"), False)]
    if full_fail:
        return "inconclusive: full baseline failed at least one configuration", False
    if no_lu_fail:
        return "keep full correction", False
    if bulk_fractional_fail:
        return "remove LU but retain source neighborhood", False
    return "all initial variants passed; run confirmation suite before removing source correction", True


def git_text(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=Path(__file__).resolve().parents[2], text=True).strip()
    except subprocess.CalledProcessError:
        return "unavailable"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path(__file__).resolve().parent / "results" / timestamp
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    validate_sources()
    ops = load_extensions(verbose=args.verbose_build)

    iso_rows, iso_curves = isolated_cases(ops)
    full_rows, full_curves = full_cases(ops)
    all_rows = iso_rows + full_rows
    decision, confirm = recommendation(all_rows)
    if confirm:
        extra = {f"fractional_{a}{b}{c}": (a, b, c) for a in (0.25, 0.75) for b in (0.25, 0.75) for c in (0.25, 0.75)}
        extra_rows, extra_curves = full_cases(ops, source_cases=extra, model_cases=("smooth",), label="confirmation")
        all_rows.extend(extra_rows)
        full_curves.update(extra_curves)
        confirmation_failed = any(not bool(row["pass"]) for row in extra_rows if row["variant"] == "bulk_only")
        decision = "keep full correction" if confirmation_failed else "remove the source correction"
    runtime_rows = benchmark(ops)
    curves = {**iso_curves, **full_curves}
    plot_curves(curves, output)
    write_csv(output / "taylor_results.csv", all_rows)
    write_csv(output / "runtime.csv", runtime_rows)
    manifest = {
        "timestamp_utc": timestamp,
        "git_head": git_text("rev-parse", "HEAD"),
        "git_status": git_text("status", "--short"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "epsilons": EPSILONS,
        "production_sha256": sha256(PRODUCTION),
        "variant_sha256": {name: sha256(path) for name, (_, path) in VARIANTS.items()},
        "cwd": os.getcwd(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = ["# Source-correction experiment", "", f"Recommendation: **{decision}**", "", "## Taylor summary", "", "| Level | Model | Source | Variant | slope | pass |", "|---|---|---|---|---:|---|"]
    seen = set()
    for row in all_rows:
        key = (row["level"], row["model"], row["source"], row["variant"])
        if key not in seen:
            seen.add(key)
            lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {float(row['representative_slope']):.3f} | {'PASS' if row['pass'] else 'FAIL'} |")
    lines.extend(["", "## Isolated relative gradient difference from full", "", "| Model | Source | Variant | relative difference |", "|---|---|---|---:|"])
    seen = set()
    for row in all_rows:
        key = (row["level"], row["model"], row["source"], row["variant"])
        if row["level"] == "isolated" and key not in seen:
            seen.add(key)
            lines.append(f"| {key[1]} | {key[2]} | {key[3]} | {float(row['relative_gradient_difference']):.6e} |")
    lines.extend(["", "## Backward timing", "", "| Variant | median ms | IQR ms | ratio to full |", "|---|---:|---:|---:|"])
    for row in runtime_rows:
        lines.append(f"| {row['variant']} | {row['median_ms']:.3f} | {row['iqr_ms']:.3f} | {row['speed_ratio_to_full']:.3f} |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nArtifacts: {output}")


if __name__ == "__main__":
    main()
