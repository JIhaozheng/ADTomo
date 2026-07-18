"""Compare adjoint backends vs LU baseline (2D x–z and 3D center-line x–z).

  ordered  — discrete ordered back-substitution (Li et al., 2013)
  0f_src   — continuous FSM, zero-flux BC, with source gradient correction
  0f       — continuous FSM, zero-flux BC, without source gradient correction
  dir      — continuous FSM, Dirichlet BC

Edit knobs below, then run: python compare_lu.py
Figures: tests/gradtest_figs/compare_lu_{2d,3d}.png
"""

import importlib
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

RUN_2D, RUN_3D = True, True
XMAX, YMAX, ZMAX = 6.0, 0.1, 3.0
H_2D, H_3D = 0.02, 0.02
V_TOP, V_BOT = 2.0, 6.0
ANOM_A, ANOM_B, ANOM_C = 1.5, 0.7, 0.7
ANOM_CX, ANOM_CY, ANOM_CZ, ANOM_V = 3.0, None, 1.5, 5.0
EVENT_X, EVENT_Y, EVENT_Z = 0.2, None, 0.2
STA_LINES = [
    (np.linspace(1.5, 5.5, 5), 0.2),
    (5.5, np.linspace(1, 2.5, 3)),
]
ADJ_WEIGHT, EPS_PRECOND = 1.0, 1e-3
OUT_DIR = Path(__file__).resolve().parent / "gradtest_figs"
DPI = 150

LABELS = ("ordered", "0f_src", "0f", "dir")
BACKENDS_2D = {
    "ordered": "eikonal2d_adj_ordered_op",
    "0f_src": "eikonal2d_adj_0f_src_op",
    "0f": "eikonal2d_adj_0f_op",
    "dir": "eikonal2d_adj_dir_op",
}
BACKENDS_3D = {
    "ordered": "eikonal3d_adj_ordered_op",
    "0f_src": "eikonal3d_adj_0f_src_op",
    "0f": "eikonal3d_adj_0f_op",
    "dir": "eikonal3d_adj_dir_op",
}


def _bw(op):
    return getattr(op, "backward_ordered", None) or op.backward


def expand_stations(lines):
    xs, zs = [], []
    for x, z in lines:
        xa, za = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        xs.append(xa.ravel())
        zs.append(za.ravel())
    return np.concatenate(xs), np.concatenate(zs)


def inject2d(shape, stations, weight):
    gu = torch.zeros(shape, dtype=torch.float64)
    for rx, rz in stations:
        i0, k0 = int(np.floor(rx)), int(np.floor(rz))
        wx, wz = rx - i0, rz - k0
        for di, wi in ((0, 1 - wx), (1, wx)):
            for dk, wk in ((0, 1 - wz), (1, wz)):
                ii, kk = i0 + di, k0 + dk
                if 0 <= ii < shape[0] and 0 <= kk < shape[1]:
                    gu[ii, kk] += weight * wi * wk
    return gu


def inject3d(shape, stations, weight):
    gu = torch.zeros(shape, dtype=torch.float64)
    for rx, ry, rz in stations:
        i0, j0, k0 = int(np.floor(rx)), int(np.floor(ry)), int(np.floor(rz))
        wx, wy, wz = rx - i0, ry - j0, rz - k0
        for di, wi in ((0, 1 - wx), (1, wx)):
            for dj, wj in ((0, 1 - wy), (1, wy)):
                for dk, wk in ((0, 1 - wz), (1, wz)):
                    ii, jj, kk = i0 + di, j0 + dj, k0 + dk
                    if 0 <= ii < shape[0] and 0 <= jj < shape[1] and 0 <= kk < shape[2]:
                        gu[ii, jj, kk] += weight * wi * wj * wk
    return gu


def to_grad_v(gf, gf_e, f, v, h):
    lam = gf / (2.0 * f * h * h + 1e-30)
    lam_e = gf_e / (2.0 * f * h * h + 1e-30)
    g = -lam / (v**3 + 1e-30)
    g_pc = -lam / (lam_e + EPS_PRECOND * float(np.max(np.abs(lam_e))) + 1e-30)
    return g, g_pc


def style_xz(ax, title, xmax, zmax):
    ax.set_title(title)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")
    ax.set_xlim(0, xmax)
    ax.set_ylim(zmax, 0)


def plot_field(ax, field, extent, event, stas, title, xmax, zmax, cmap="viridis", vmin=None, vmax=None):
    kw = dict(origin="upper", aspect="auto", cmap=cmap, extent=extent)
    if vmin is not None:
        kw.update(vmin=vmin, vmax=vmax)
    im = ax.imshow(field.T, **kw)
    ax.scatter([event[0]], [event[1]], c="k", marker="*", s=90, zorder=3, label="event")
    ax.scatter([p[0] for p in stas], [p[1] for p in stas], c="k", marker="^", s=45, zorder=3, label="station")
    style_xz(ax, title, xmax, zmax)
    return plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_hist(ax, abs_d, label):
    log_d = np.log10(np.maximum(abs_d, 1e-30))
    lo, hi = max(float(log_d.min()), -30.0), max(float(log_d.max()), float(log_d.min()) + 1e-6)
    st_max, st_mean = float(abs_d.max()), float(abs_d.mean())
    st_rms = float(np.sqrt(np.mean(abs_d**2)))
    ax.hist(log_d, bins=np.linspace(lo, hi, 40), edgecolor="black", alpha=0.75)
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xticks(np.linspace(lo, hi, 5))
    ax.set_xlabel(rf"$\log_{{10}}|g-g_{{\mathrm{{LU}}}}|$ (max$={st_max:.2e}$)")
    ax.set_ylabel("count")
    ax.set_title(f"{label} residual hist")
    ax.text(0.98, 0.98, f"max={st_max:.2e}\nmean={st_mean:.2e}\nrms={st_rms:.2e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    return st_max


def run_backend(op, f, h, src, gu, gu_e, v, gf_ref=None):
    bw = _bw(op)
    u = op.forward(f, h, *src)
    t0 = time.perf_counter()
    gf = bw(gu, u, f, h, *src).detach().numpy()
    t_raw = time.perf_counter() - t0
    t0 = time.perf_counter()
    gf_e = bw(gu_e, u, f, h, *src).detach().numpy()
    t_pc = t_raw + (time.perf_counter() - t0)
    fn = f.detach().numpy()
    g, g_pc = to_grad_v(gf, gf_e, fn, v, h)
    dmax = float(np.max(np.abs(gf - gf_ref))) if gf_ref is not None else 0.0
    return gf, g, g_pc, t_raw, t_pc, dmax


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sta_x, sta_z = expand_stations(STA_LINES)

    if RUN_2D:
        print(f"\n{'=' * 48}\n2D x–z\n{'=' * 48}")
        h = H_2D
        nx, nz = int(round(XMAX / h)) + 1, int(round(ZMAX / h)) + 1
        z = np.arange(nz) * h
        v_init = np.broadcast_to((V_TOP + (V_BOT - V_TOP) * (z / ZMAX))[None, :], (nx, nz)).copy()
        v_true = v_init.copy()
        X, Z = np.meshgrid(np.arange(nx) * h, z, indexing="ij")
        v_true[((X - ANOM_CX) / ANOM_A) ** 2 + ((Z - ANOM_CZ) / ANOM_C) ** 2 <= 1] = ANOM_V

        sx, sz = EVENT_X / h, EVENT_Z / h
        stations = [(xi / h, zi / h) for xi, zi in zip(sta_x, sta_z)]
        stas_km = [(p[0] * h, p[1] * h) for p in stations]
        f = torch.from_numpy(1.0 / v_init).contiguous()
        gu = inject2d((nx, nz), stations, ADJ_WEIGHT)
        gu_e = inject2d((nx, nz), stations, 1.0)

        op_lu = importlib.import_module("eikonal2d_op")
        gf_lu, g_lu, g_lu_pc, t_lu, t_lu_pc, _ = run_backend(
            op_lu, f, h, (float(sx), float(sz)), gu, gu_e, v_init, None
        )
        print(f"  LU         raw={t_lu:.4f} s  precond={t_lu_pc:.4f} s")

        extent = [0, (nx - 1) * h, (nz - 1) * h, 0]
        fig, axes = plt.subplots(1 + len(LABELS), 3, figsize=(15, 2.6 * (1 + len(LABELS))),
                                 constrained_layout=True)
        cb = plot_field(axes[0, 0], v_true, extent, (EVENT_X, EVENT_Z), stas_km,
                        "true model $v$", XMAX, ZMAX, cmap="viridis_r", vmin=V_TOP, vmax=V_BOT)
        cb.set_label("v (km/s)")
        axes[0, 0].legend(loc="upper right", fontsize=10)
        plot_field(axes[0, 1], g_lu, extent, (EVENT_X, EVENT_Z), stas_km,
                   rf"$\partial\mathcal{{J}}/\partial v$" + f"\n{t_lu:.4f} s", XMAX, ZMAX)
        plot_field(axes[0, 2], g_lu_pc, extent, (EVENT_X, EVENT_Z), stas_km,
                   r"preconditioned $\partial\mathcal{J}/\partial v$" + f"\n{t_lu_pc:.4f} s", XMAX, ZMAX)

        for row, name in enumerate(LABELS, 1):
            op = importlib.import_module(BACKENDS_2D[name])
            gf, g, g_pc, t_raw, t_pc, dmax = run_backend(
                op, f, h, (float(sx), float(sz)), gu, gu_e, v_init, gf_lu
            )
            print(f"  {name:10s}  raw={t_raw:.4f} s  precond={t_pc:.4f} s  max|Δ∂J/∂f|={dmax:.3e}")
            plot_hist(axes[row, 0], np.abs(gf - gf_lu).ravel(), name)
            plot_field(axes[row, 1], g, extent, (EVENT_X, EVENT_Z), stas_km, f"{t_raw:.4f} s", XMAX, ZMAX)
            plot_field(axes[row, 2], g_pc, extent, (EVENT_X, EVENT_Z), stas_km, f"{t_pc:.4f} s", XMAX, ZMAX)

        path = OUT_DIR / "compare_lu_2d.png"
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        print(f"  -> {path}")

    if RUN_3D:
        print(f"\n{'=' * 48}\n3D center-line x–z\n{'=' * 48}")
        h = H_3D
        nx = int(round(XMAX / h)) + 1
        ny = int(round(YMAX / h)) + 1
        nz = int(round(ZMAX / h)) + 1
        ky = (ny - 1) // 2
        ey = EVENT_Y if EVENT_Y is not None else ky * h
        acy = ANOM_CY if ANOM_CY is not None else ey

        z = np.arange(nz) * h
        v_init = np.broadcast_to((V_TOP + (V_BOT - V_TOP) * (z / ZMAX))[None, None, :], (nx, ny, nz)).copy()
        v_true = v_init.copy()
        X, Y, Z = np.meshgrid(np.arange(nx) * h, np.arange(ny) * h, z, indexing="ij")
        v_true[
            ((X - ANOM_CX) / ANOM_A) ** 2 + ((Y - acy) / ANOM_B) ** 2 + ((Z - ANOM_CZ) / ANOM_C) ** 2 <= 1
        ] = ANOM_V

        sx, sy, sz = EVENT_X / h, ey / h, EVENT_Z / h
        stations = [(xi / h, sy, zi / h) for xi, zi in zip(sta_x, sta_z)]
        stas_km = [(p[0] * h, p[2] * h) for p in stations]
        f = torch.from_numpy(1.0 / v_init).contiguous()
        gu = inject3d((nx, ny, nz), stations, ADJ_WEIGHT)
        gu_e = inject3d((nx, ny, nz), stations, 1.0)
        src = (float(sx), float(sy), float(sz))

        op_lu = importlib.import_module("eikonal3d_op")
        gf_lu, g_lu, g_lu_pc, t_lu, t_lu_pc, _ = run_backend(
            op_lu, f, h, src, gu, gu_e, v_init, None
        )
        print(f"  LU         raw={t_lu:.4f} s  precond={t_lu_pc:.4f} s")
        gf_lu_s, g_lu_s, g_lu_pc_s = gf_lu[:, ky, :], g_lu[:, ky, :], g_lu_pc[:, ky, :]

        extent = [0, (nx - 1) * h, (nz - 1) * h, 0]
        fig, axes = plt.subplots(1 + len(LABELS), 3, figsize=(15, 2.6 * (1 + len(LABELS))),
                                 constrained_layout=True)
        cb = plot_field(axes[0, 0], v_true[:, ky, :], extent, (EVENT_X, EVENT_Z), stas_km,
                        rf"true $v$ ($y$={ey:.2f})", XMAX, ZMAX, vmin=V_TOP, vmax=V_BOT)
        cb.set_label("v (km/s)")
        axes[0, 0].legend(loc="upper right", fontsize=10)
        plot_field(axes[0, 1], g_lu_s, extent, (EVENT_X, EVENT_Z), stas_km,
                   rf"$\partial\mathcal{{J}}/\partial v$" + f"\n{t_lu:.4f} s", XMAX, ZMAX)
        plot_field(axes[0, 2], g_lu_pc_s, extent, (EVENT_X, EVENT_Z), stas_km,
                   r"preconditioned $\partial\mathcal{J}/\partial v$" + f"\n{t_lu_pc:.4f} s", XMAX, ZMAX)

        for row, name in enumerate(LABELS, 1):
            op = importlib.import_module(BACKENDS_3D[name])
            gf, g, g_pc, t_raw, t_pc, _ = run_backend(op, f, h, src, gu, gu_e, v_init, gf_lu)
            gf_s, g_s, g_pc_s = gf[:, ky, :], g[:, ky, :], g_pc[:, ky, :]
            dmax = float(np.max(np.abs(gf_s - gf_lu_s)))
            print(f"  {name:10s}  raw={t_raw:.4f} s  precond={t_pc:.4f} s  max|Δ∂J/∂f|={dmax:.3e}")
            plot_hist(axes[row, 0], np.abs(gf_s - gf_lu_s).ravel(), name)
            plot_field(axes[row, 1], g_s, extent, (EVENT_X, EVENT_Z), stas_km, f"{t_raw:.4f} s", XMAX, ZMAX)
            plot_field(axes[row, 2], g_pc_s, extent, (EVENT_X, EVENT_Z), stas_km, f"{t_pc:.4f} s", XMAX, ZMAX)

        path = OUT_DIR / "compare_lu_3d.png"
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        print(f"  -> {path}")

    print(f"\nFigures under {OUT_DIR}")
