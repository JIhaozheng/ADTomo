// 2D continuous zero-flux FSM adjoint no source-corner (src) correction

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

static constexpr double FSM_INF = 1.0e20;

static inline int gid(int i, int j, int ny) { return i * ny + j; }

static double godunov_update_2d(double tx, double ty, double s, double dx, double dy) {
    if (tx >= FSM_INF && ty >= FSM_INF) return FSM_INF;
    if (tx >= FSM_INF) return ty + s * dy;
    if (ty >= FSM_INF) return tx + s * dx;
    double t1 = tx + s * dx;
    if (t1 <= ty) return t1;
    double t2 = ty + s * dy;
    if (t2 <= tx) return t2;
    const double a = 1.0 / (dx * dx) + 1.0 / (dy * dy);
    const double b = -2.0 * tx / (dx * dx) - 2.0 * ty / (dy * dy);
    const double c = tx * tx / (dx * dx) + ty * ty / (dy * dy) - s * s;
    const double disc = b * b - 4.0 * a * c;
    if (disc < 0.0) return std::min(tx + s * dx, ty + s * dy);
    return std::max((-b + std::sqrt(disc)) / (2.0 * a), std::max(tx, ty));
}

static void sweep_fsm_inf(double *u, const double *f, int nx, int ny, double h,
                          int ix0, int iy0, int ix1, int iy1,
                          double tol = 1e-6, int max_iter = 200) {
    const int nn = nx * ny;
    auto T_at = [&](int i, int j) -> double {
        if (i < 0 || i >= nx || j < 0 || j >= ny) return FSM_INF;
        return u[gid(i, j, ny)];
    };
    auto is_sc = [&](int i, int j) {
        return (i == ix0 && j == iy0) || (i == ix1 && j == iy0) ||
               (i == ix0 && j == iy1) || (i == ix1 && j == iy1);
    };
    std::vector<double> old(nn);
    for (int it = 0; it < max_iter; ++it) {
        old.assign(u, u + nn);
        auto sweep = [&](int x0, int x1, int xd, int y0, int y1, int yd, auto left, auto down) {
            for (int j = y0; j != y1; j += yd)
                for (int i = x0; i != x1; i += xd) {
                    if (is_sc(i, j)) continue;
                    const int k = gid(i, j, ny);
                    u[k] = std::min(u[k], godunov_update_2d(left(j, i), down(j, i), f[k], h, h));
                }
        };
        sweep(0, nx, 1, 0, ny, 1,
              [&](int j, int i) { return i > 0 ? T_at(i - 1, j) : FSM_INF; },
              [&](int j, int i) { return j > 0 ? T_at(i, j - 1) : FSM_INF; });
        sweep(0, nx, 1, ny - 1, -1, -1,
              [&](int j, int i) { return i > 0 ? T_at(i - 1, j) : FSM_INF; },
              [&](int j, int i) { return j < ny - 1 ? T_at(i, j + 1) : FSM_INF; });
        sweep(nx - 1, -1, -1, 0, ny, 1,
              [&](int j, int i) { return i < nx - 1 ? T_at(i + 1, j) : FSM_INF; },
              [&](int j, int i) { return j > 0 ? T_at(i, j - 1) : FSM_INF; });
        sweep(nx - 1, -1, -1, ny - 1, -1, -1,
              [&](int j, int i) { return i < nx - 1 ? T_at(i + 1, j) : FSM_INF; },
              [&](int j, int i) { return j < ny - 1 ? T_at(i, j + 1) : FSM_INF; });
        double err = 0.0;
        for (int k = 0; k < nn; ++k) err = std::max(err, std::fabs(u[k] - old[k]));
        if (err < tol) break;
    }
}

static void forward(double *u, const double *f, int m, int n, double h, double x, double y) {
    const int nx = m + 1, ny = n + 1, nn = nx * ny;
    int ix0 = std::max(0, std::min(static_cast<int>(std::floor(x)), nx - 2));
    int iy0 = std::max(0, std::min(static_cast<int>(std::floor(y)), ny - 2));
    int ix1 = ix0 + 1, iy1 = iy0 + 1;
    for (int k = 0; k < nn; ++k) u[k] = FSM_INF;

    double f00 = f[gid(ix0, iy0, ny)], f10 = f[gid(ix1, iy0, ny)];
    double f01 = f[gid(ix0, iy1, ny)], f11 = f[gid(ix1, iy1, ny)];
    double wx = x - ix0, wy = y - iy0;
    double fsrc = (1 - wx) * (1 - wy) * f00 + wx * (1 - wy) * f10 + (1 - wx) * wy * f01 + wx * wy * f11;
    double fmid = 0.25 * (f00 + f10 + f01 + f11);
    auto set_corner = [&](int ii, int jj, double fc) {
        double d = std::sqrt((x - ii) * (x - ii) + (y - jj) * (y - jj)) * h;
        u[gid(ii, jj, ny)] = (d / 6.0) * (fsrc + 4.0 * fmid + fc);
    };
    set_corner(ix0, iy0, f00);
    set_corner(ix1, iy0, f10);
    set_corner(ix0, iy1, f01);
    set_corner(ix1, iy1, f11);
    sweep_fsm_inf(u, f, nx, ny, h, ix0, iy0, ix1, iy1);
}

static inline bool is_source_corner(int i, int j, int ix0, int iy0, int ix1, int iy1) {
    return (i == ix0 || i == ix1) && (j == iy0 || j == iy1);
}

static void upwind_split(double a, double &am, double &ap) {
    am = (a - std::fabs(a)) * 0.5;
    ap = (a + std::fabs(a)) * 0.5;
}

// 2D adjderivonsource: normal-projected ∇(-T) at source-box corners.
template <typename TAt>
static void adjderivonsource(const TAt &T_at, int nx, int ny, int i, int j, double h,
                             double sx, double sy, int ix0, int iy0, int ix1, int iy1,
                             double &aback, double &aforw, double &bback, double &bforw) {
    const double Ti = T_at(i, j);
    const double distPSx = std::fabs(sx - static_cast<double>(i)) * h;
    const double distPSy = std::fabs(sy - static_cast<double>(j)) * h;
    const double dist2src = std::hypot(sx - i, sy - j) * h;
    aback = aforw = bback = bforw = 0.0;
    if (dist2src < 1e-15) return;

    const double thx = Ti * distPSy / dist2src;
    const double thy = Ti * distPSx / dist2src;
    auto is_sc = [&](int ii, int jj) { return is_source_corner(ii, jj, ix0, iy0, ix1, iy1); };

    if (i < nx - 1 && is_sc(i + 1, j)) {
        if (i > 0) aback = -(Ti - T_at(i - 1, j)) / h;
        aforw = (distPSx < 1e-15) ? 0.0 : -(thx - Ti) / distPSx;
    } else if (i > 0 && is_sc(i - 1, j)) {
        aback = (distPSx < 1e-15) ? 0.0 : -(Ti - thx) / distPSx;
        if (i < nx - 1) aforw = -(T_at(i + 1, j) - Ti) / h;
    } else {
        if (i > 0) aback = -(Ti - T_at(i - 1, j)) / h;
        if (i < nx - 1) aforw = -(T_at(i + 1, j) - Ti) / h;
    }

    if (j < ny - 1 && is_sc(i, j + 1)) {
        if (j > 0) bback = -(Ti - T_at(i, j - 1)) / h;
        bforw = (distPSy < 1e-15) ? 0.0 : -(thy - Ti) / distPSy;
    } else if (j > 0 && is_sc(i, j - 1)) {
        bback = (distPSy < 1e-15) ? 0.0 : -(Ti - thy) / distPSy;
        if (j < ny - 1) bforw = -(T_at(i, j + 1) - Ti) / h;
    } else {
        if (j > 0) bback = -(Ti - T_at(i, j - 1)) / h;
        if (j < ny - 1) bforw = -(T_at(i, j + 1) - Ti) / h;
    }
}

static double adjoint_stencil_0f(const double *T, const double *lam, const double *delta,
                                 int nx, int ny, int i, int j, double h,
                                 double sx, double sy) {
    auto T_at = [&](int ii, int jj) -> double {
        if (ii < 0 || ii >= nx || jj < 0 || jj >= ny) return 0.0;
        return T[gid(ii, jj, ny)];
    };
    auto L_at = [&](int ii, int jj) -> double {
        if (ii < 0 || ii >= nx || jj < 0 || jj >= ny) return 0.0;
        return lam[gid(ii, jj, ny)];
    };
    int ix0 = std::max(0, std::min((int)std::floor(sx), nx - 2));
    int iy0 = std::max(0, std::min((int)std::floor(sy), ny - 2));
    int ix1 = ix0 + 1, iy1 = iy0 + 1;

    double a1m = 0, a1p = 0, a2m = 0, a2p = 0, b1m = 0, b1p = 0, b2m = 0, b2p = 0;
    if (is_source_corner(i, j, ix0, iy0, ix1, iy1)) {
        double aback, aforw, bback, bforw;
        adjderivonsource(T_at, nx, ny, i, j, h, sx, sy, ix0, iy0, ix1, iy1,
                         aback, aforw, bback, bforw);
        upwind_split(aback, a1m, a1p);
        upwind_split(aforw, a2m, a2p);
        upwind_split(bback, b1m, b1p);
        upwind_split(bforw, b2m, b2p);
    } else {
        if (i > 0) {
            double a1 = -(T_at(i, j) - T_at(i - 1, j)) / h;
            upwind_split(a1, a1m, a1p);
        }
        if (i < nx - 1) {
            double a2 = -(T_at(i + 1, j) - T_at(i, j)) / h;
            upwind_split(a2, a2m, a2p);
        }
        if (j > 0) {
            double b1 = -(T_at(i, j) - T_at(i, j - 1)) / h;
            upwind_split(b1, b1m, b1p);
        }
        if (j < ny - 1) {
            double b2 = -(T_at(i, j + 1) - T_at(i, j)) / h;
            upwind_split(b2, b2m, b2p);
        }
    }
    const double coe = (a2p - a1m) / h + (b2p - b1m) / h;
    if (std::fabs(coe) < 1e-15) return 0.0;
    const double hadj = (a1p * L_at(i - 1, j) - a2m * L_at(i + 1, j)) / h +
                        (b1p * L_at(i, j - 1) - b2m * L_at(i, j + 1)) / h;
    return (delta[gid(i, j, ny)] + hadj) / coe;
}

static void solve_adjoint_fsm_0f(double *lam, const double *T, const double *delta,
                                 int nx, int ny, double h, double sx, double sy,
                                 int max_iter = 200, double tol = 1e-6) {
    const int nn = nx * ny;
    std::fill(lam, lam + nn, 0.0);
    std::vector<double> old(nn);
    for (int it = 0; it < max_iter; ++it) {
        old.assign(lam, lam + nn);
        for (int sx_d : {1, -1})
            for (int sy_d : {1, -1}) {
                auto I = std::make_tuple(sx_d == 1 ? 0 : nx - 1, sx_d == 1 ? nx : -1, sx_d);
                auto J = std::make_tuple(sy_d == 1 ? 0 : ny - 1, sy_d == 1 ? ny : -1, sy_d);
                for (int i = std::get<0>(I); i != std::get<1>(I); i += std::get<2>(I))
                    for (int j = std::get<0>(J); j != std::get<1>(J); j += std::get<2>(J))
                        lam[gid(i, j, ny)] =
                            adjoint_stencil_0f(T, lam, delta, nx, ny, i, j, h, sx, sy);
            }
        double err = 0.0;
        for (int k = 0; k < nn; ++k) err = std::max(err, std::fabs(lam[k] - old[k]));
        if (err < tol) break;
    }
}

static void backward(double *grad_f, const double *grad_u, const double *u, const double *f,
                     int m, int n, double h, double x, double y) {
    const int nx = m + 1, ny = n + 1, nn = nx * ny;
    const double area = h * h;
    const double lam_scale = 0.5;
    std::vector<double> delta(nn), lambda(nn);
    for (int i = 0; i < nn; ++i) delta[i] = grad_u[i] / area;
    solve_adjoint_fsm_0f(lambda.data(), u, delta.data(), nx, ny, h, x, y);
    for (int i = 0; i < nn; ++i) grad_f[i] = (lambda[i] * lam_scale) * 2.0 * f[i] * area;
}

torch::Tensor eikonal_forward(torch::Tensor f, double h, double x, double y) {
    TORCH_CHECK(f.dim() == 2, "f should be 2D");
    TORCH_CHECK(f.is_contiguous(), "f should be contiguous");
    auto m = f.size(0) - 1, n = f.size(1) - 1;
    auto u = torch::zeros_like(f);
    forward(u.data_ptr<double>(), f.data_ptr<double>(), m, n, h, x, y);
    return u;
}

torch::Tensor eikonal_backward(torch::Tensor grad_u, torch::Tensor u, torch::Tensor f,
                               double h, double x, double y) {
    TORCH_CHECK(grad_u.dim() == 2, "grad_u should be 2D");
    TORCH_CHECK(grad_u.sizes() == u.sizes(), "grad_u and u should have the same size");
    TORCH_CHECK(grad_u.is_contiguous(), "grad_u should be contiguous");
    auto m = f.size(0) - 1, n = f.size(1) - 1;
    auto grad_f = torch::zeros_like(f);
    backward(grad_f.data_ptr<double>(), grad_u.data_ptr<double>(), u.data_ptr<double>(),
             f.data_ptr<double>(), m, n, h, x, y);
    return grad_f;
}

torch::Tensor eikonal_solve_adjoint(torch::Tensor T, torch::Tensor delta, double h,
                                    double x, double y) {
    TORCH_CHECK(T.dim() == 2 && delta.dim() == 2, "T and delta must be 2D");
    TORCH_CHECK(T.sizes() == delta.sizes(), "T and delta must have the same shape");
    TORCH_CHECK(T.is_contiguous() && delta.is_contiguous(), "T and delta must be contiguous");
    int nx = T.size(0), ny = T.size(1);
    auto lambda = torch::zeros_like(T);
    solve_adjoint_fsm_0f(lambda.data_ptr<double>(), T.data_ptr<double>(),
                         delta.data_ptr<double>(), nx, ny, h, x, y);
    return lambda;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &eikonal_forward, "2D forward");
    m.def("backward", &eikonal_backward, "2D zero-flux FSM adjoint");
    m.def("solve_adjoint", &eikonal_solve_adjoint, "2D zero-flux FSM adjoint");
}
