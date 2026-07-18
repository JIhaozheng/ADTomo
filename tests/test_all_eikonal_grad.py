"""Taylor gradient test for all eikonal adjoint backends (2D and 3D).
Merge 2D (top) and 3D (bottom) into single figure: gradtest_combined_2d3d.png
All logic inside __main__ without split compute/plot functions
"""

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_DIR = Path(__file__).resolve().parent / "gradtest_figs"

BACKENDS_2D = {
    "LU": "eikonal2d_op",
    "ordered": "eikonal2d_adj_ordered_op",
    "0f_src": "eikonal2d_adj_0f_src_op",
    "global": "eikonal2d_adj_global_op",
    "dir": "eikonal2d_adj_dir_op",
    "0f": "eikonal2d_adj_0f_op",
}

BACKENDS_3D = {
    "LU": "eikonal3d_op",
    "ordered": "eikonal3d_adj_ordered_op",
    "0f_src": "eikonal3d_adj_0f_src_op",
    "global": "eikonal3d_adj_global_op",
    "dir": "eikonal3d_adj_dir_op",
    "0f": "eikonal3d_adj_0f_op",
}

# grid / source / step
H = 0.5
GS = [10 ** (-i) for i in range(1, 6)]
M2, N2 = 2, 3
IX, JX = 0.0, 0.0
M3, N3, L3 = 3, 3, 2
X, Y, Z = 0.0, 0.0, 0.0


def make_eikonal2d(op):
    bw = getattr(op, "backward_ordered", None) or op.backward

    class Eikonal2DFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, f, h, ix, jx):
            u = op.forward(f, h, ix, jx)
            ctx.save_for_backward(u, f)
            ctx.h, ctx.ix, ctx.jx = h, ix, jx
            return u

        @staticmethod
        def backward(ctx, grad_output):
            u, f = ctx.saved_tensors
            grad_f = bw(grad_output.contiguous(), u, f, ctx.h, ctx.ix, ctx.jx)
            return grad_f, None, None, None

    class Eikonal2D(torch.nn.Module):
        def __init__(self, h, ix, jx):
            super(Eikonal2D, self).__init__()
            self.h, self.ix, self.jx = h, ix, jx

        def forward(self, f):
            return Eikonal2DFunction.apply(f, self.h, self.ix, self.jx)

    return Eikonal2D


def make_eikonal3d(op):
    bw = getattr(op, "backward_ordered", None) or op.backward

    class Eikonal3DFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, f, h, x, y, z):
            u = op.forward(f, h, x, y, z)
            ctx.save_for_backward(u, f)
            ctx.h, ctx.x, ctx.y, ctx.z = h, x, y, z
            return u

        @staticmethod
        def backward(ctx, grad_output):
            u, f = ctx.saved_tensors
            grad_f = bw(grad_output.contiguous(), u, f, ctx.h, ctx.x, ctx.y, ctx.z)
            return grad_f, None, None, None, None

    class Eikonal3D(torch.nn.Module):
        def __init__(self, h, x, y, z):
            super(Eikonal3D, self).__init__()
            self.h, self.x, self.y, self.z = h, x, y, z

        def forward(self, f):
            return Eikonal3DFunction.apply(
                f, self.h, float(self.x), float(self.y), float(self.z)
            )

    return Eikonal3D


def taylor_test(eikonal_solver, f, v_, gs_):
    """Same procedure as test_eikonal{2,3}d_grad_op.py."""
    m_ = f

    def scalar_function(ff):
        u = eikonal_solver(ff)
        return torch.sum(u)

    y_ = scalar_function(m_)
    y_.backward()
    dy_ = f.grad
    y_ = y_.item()

    ms_, ys_, s_, w_ = [], [], [], []
    for g_ in gs_:
        ms_.append(m_ + g_ * v_)
        ys_.append(scalar_function(ms_[-1]).item())
        s_.append(ys_[-1] - y_)
        w_.append(s_[-1] - g_ * torch.sum(v_ * dy_).item())

    return s_, w_


def plot_one(ax, gs_, s_, w_, title):
    ax.loglog(gs_, np.abs(s_), "*-", label="finite difference")
    ax.loglog(gs_, np.abs(w_), "+-", label="automatic differentiation")
    ax.loglog(
        gs_,
        [g**2 * 0.5 * abs(w_[0]) / gs_[0] ** 2 for g in gs_],
        "--",
        label=r"$\mathcal{O}(\gamma^2)$",
    )
    ax.loglog(
        gs_,
        [g * 0.5 * abs(s_[0]) / gs_[0] for g in gs_],
        "--",
        label=r"$\mathcal{O}(\gamma)$",
    )
    ax.invert_xaxis()
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("Error")
    ax.set_title(title)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------- Compute 2D data --------------------------
    print("\n2D gradient test compute")
    lab2d, res2d = [], []
    for name, modname in BACKENDS_2D.items():
        try:
            op = importlib.import_module(modname)
        except Exception as e:
            print(f"  {name}: IMPORT FAIL ({e})")
            continue
        torch.manual_seed(0)
        f_ = torch.ones((M2, N2), dtype=torch.float64)
        f = torch.nn.Parameter(f_, requires_grad=True)
        solver = make_eikonal2d(op)(H, IX, JX)
        v_ = torch.randn(M2, N2, dtype=torch.float64)
        s_, w_ = taylor_test(solver, f, v_, GS)
        lab2d.append(name)
        res2d.append((s_, w_))
        print(f"  {name}: done")

    # -------------------------- Compute 3D data --------------------------
    print("\n3D gradient test compute")
    lab3d, res3d = [], []
    for name, modname in BACKENDS_3D.items():
        try:
            op = importlib.import_module(modname)
        except Exception as e:
            print(f"  {name}: IMPORT FAIL ({e})")
            continue
        torch.manual_seed(0)
        f_ = torch.ones((M3, N3, L3), dtype=torch.float64)
        f = torch.nn.Parameter(f_, requires_grad=True)
        solver = make_eikonal3d(op)(H, X, Y, Z)
        v_ = torch.randn(M3, N3, L3, dtype=torch.float64)
        s_, w_ = taylor_test(solver, f, v_, GS)
        lab3d.append(name)
        res3d.append((s_, w_))
        print(f"  {name}: done")

    # -------------------------- Draw combined figure --------------------------
    ncol = 3
    nrow2d = (len(lab2d) + ncol - 1) // ncol
    nrow3d = (len(lab3d) + ncol - 1) // ncol
    total_rows = nrow2d + nrow3d
    fig_w = 4.2 * ncol
    fig_h = 3.6 * total_rows
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)

    # Top: 2D subplots
    for i in range(nrow2d):
        for j in range(ncol):
            idx = i * ncol + j
            ax = fig.add_subplot(total_rows, ncol, idx + 1)
            if idx < len(lab2d):
                plot_one(ax, GS, res2d[idx][0], res2d[idx][1], f"2D {lab2d[idx]}")
            else:
                ax.axis("off")

    # Bottom: 3D subplots, offset by all 2D axes
    offset = nrow2d * ncol
    for i in range(nrow3d):
        for j in range(ncol):
            idx = i * ncol + j
            ax_pos = offset + i * ncol + j + 1
            ax = fig.add_subplot(total_rows, ncol, ax_pos)
            if idx < len(lab3d):
                plot_one(ax, GS, res3d[idx][0], res3d[idx][1], f"3D {lab3d[idx]}")
            else:
                ax.axis("off")

    fig.suptitle("Taylor Gradient Test: 2D (Top) & 3D (Bottom) Eikonal Adjoint Backends", fontsize=14)
    out_path = OUT_DIR / "gradtest_test.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"\nCombined figure saved: {out_path}")
