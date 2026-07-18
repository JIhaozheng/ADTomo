// 2D continuous zero-flux FSM adjoint + source-corner (src) correction (same as Eikonal2D.cpp).

#include <torch/extension.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <vector>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> Trip;

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

static double adjoint_stencil_0f_src(const double *T, const double *lam, const double *delta,
                                     int nx, int ny, int i, int j, double h,
                                     double sx, double sy, bool use_source_fix) {
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
    if (use_source_fix && is_source_corner(i, j, ix0, iy0, ix1, iy1)) {
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

static void solve_adjoint_fsm_0f_src(double *lam, const double *T, const double *delta,
                                     int nx, int ny, double h, double sx, double sy,
                                     bool use_source_fix, int max_iter = 200, double tol = 1e-6) {
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
                        lam[gid(i, j, ny)] = adjoint_stencil_0f_src(
                            T, lam, delta, nx, ny, i, j, h, sx, sy, use_source_fix);
            }
        double err = 0.0;
        for (int k = 0; k < nn; ++k) err = std::max(err, std::fabs(lam[k] - old[k]));
        if (err < tol) break;
    }
}

static void apply_source_simpson_grad_from_res(double *grad_f, const double *res,
                                               int m, int n, double h, double x, double y) {
    int ix0 = std::max(0, std::min((int)std::floor(x), m));
    int jx0 = std::max(0, std::min((int)std::floor(y), n));
    int ix1 = ix0 + 1, jx1 = jx0 + 1;
    const int ny = n + 1;

    double wx = x - ix0, wy = y - jx0;
    double w00 = (1 - wx) * (1 - wy), w10 = wx * (1 - wy);
    double w01 = (1 - wx) * wy, w11 = wx * wy;

    double r00 = res[gid(ix0, jx0, ny)], r10 = res[gid(ix1, jx0, ny)];
    double r01 = res[gid(ix0, jx1, ny)], r11 = res[gid(ix1, jx1, ny)];
    double d00 = std::sqrt((x - ix0) * (x - ix0) + (y - jx0) * (y - jx0)) * h;
    double d10 = std::sqrt((x - ix1) * (x - ix1) + (y - jx0) * (y - jx0)) * h;
    double d01 = std::sqrt((x - ix0) * (x - ix0) + (y - jx1) * (y - jx1)) * h;
    double d11 = std::sqrt((x - ix1) * (x - ix1) + (y - jx1) * (y - jx1)) * h;

    grad_f[gid(ix0, jx0, ny)] =
        (r00 * d00 * (w00 + 2) + r10 * d10 * (w00 + 1) + r01 * d01 * (w00 + 1) + r11 * d11 * (w00 + 1)) / 6.0;
    grad_f[gid(ix1, jx0, ny)] =
        (r00 * d00 * (w10 + 1) + r10 * d10 * (w10 + 2) + r01 * d01 * (w10 + 1) + r11 * d11 * (w10 + 1)) / 6.0;
    grad_f[gid(ix0, jx1, ny)] =
        (r00 * d00 * (w01 + 1) + r10 * d10 * (w01 + 1) + r01 * d01 * (w01 + 2) + r11 * d11 * (w01 + 1)) / 6.0;
    grad_f[gid(ix1, jx1, ny)] =
        (r00 * d00 * (w11 + 1) + r10 * d10 * (w11 + 1) + r01 * d01 * (w11 + 1) + r11 * d11 * (w11 + 2)) / 6.0;
}

// Assemble one Godunov Jacobian row (same stencil as Eikonal2D.cpp).
static void append_godunov_row(std::vector<Trip> &triplets, const double *u, int m, int n,
                               int i, int j, int ix0, int jx0, int ix1, int jx1) {
    const int ny = n + 1;
    int this_id = gid(i, j, ny);
    if (is_source_corner(i, j, ix0, jx0, ix1, jx1)) {
        triplets.emplace_back(this_id, this_id, 1.0);
        return;
    }
    auto U = [&](int ii, int jj) { return u[gid(ii, jj, ny)]; };

    if (i == 0) {
        if (U(i, j) > U(i + 1, j)) {
            triplets.emplace_back(this_id, this_id, 2.0 * (U(i, j) - U(i + 1, j)));
            triplets.emplace_back(this_id, gid(i + 1, j, ny), 2.0 * (U(i + 1, j) - U(i, j)));
        }
    } else if (i == m) {
        if (U(i, j) > U(i - 1, j)) {
            triplets.emplace_back(this_id, this_id, 2.0 * (U(i, j) - U(i - 1, j)));
            triplets.emplace_back(this_id, gid(i - 1, j, ny), 2.0 * (U(i - 1, j) - U(i, j)));
        }
    } else {
        double a = U(i + 1, j) > U(i - 1, j) ? U(i - 1, j) : U(i + 1, j);
        if (U(i, j) > a) {
            triplets.emplace_back(this_id, this_id, 2.0 * (U(i, j) - a));
            int nb = U(i + 1, j) > U(i - 1, j) ? gid(i - 1, j, ny) : gid(i + 1, j, ny);
            triplets.emplace_back(this_id, nb, 2.0 * (a - U(i, j)));
        }
    }

    if (j == 0) {
        if (U(i, j) > U(i, j + 1)) {
            triplets.emplace_back(this_id, this_id, 2.0 * (U(i, j) - U(i, j + 1)));
            triplets.emplace_back(this_id, gid(i, j + 1, ny), 2.0 * (U(i, j + 1) - U(i, j)));
        }
    } else if (j == n) {
        if (U(i, j) > U(i, j - 1)) {
            triplets.emplace_back(this_id, this_id, 2.0 * (U(i, j) - U(i, j - 1)));
            triplets.emplace_back(this_id, gid(i, j - 1, ny), 2.0 * (U(i, j - 1) - U(i, j)));
        }
    } else {
        double b = U(i, j + 1) > U(i, j - 1) ? U(i, j - 1) : U(i, j + 1);
        if (U(i, j) > b) {
            triplets.emplace_back(this_id, this_id, 2.0 * (U(i, j) - b));
            int nb = U(i, j + 1) > U(i, j - 1) ? gid(i, j - 1, ny) : gid(i, j + 1, ny);
            triplets.emplace_back(this_id, nb, 2.0 * (b - U(i, j)));
        }
    }
}

static void patch_lu_corner_res_2d(double *res_out, const double *grad_u, const double *u,
                                   const double *lam_scaled, int m, int n, double x, double y,
                                   int radius = 1) {
    const int nx = m + 1, ny = n + 1, nn = nx * ny;
    int ix0 = std::max(0, std::min((int)std::floor(x), m));
    int jx0 = std::max(0, std::min((int)std::floor(y), n));
    int ix1 = std::min(ix0 + 1, m), jx1 = std::min(jx0 + 1, n);

    int i_lo = std::max(0, ix0 - radius);
    int i_hi = std::min(m, ix1 + radius);
    int j_lo = std::max(0, jx0 - radius);
    int j_hi = std::min(n, jx1 + radius);

    std::memcpy(res_out, lam_scaled, sizeof(double) * nn);

    std::vector<int> unknown;
    unknown.reserve(4);
    for (int i : {ix0, ix1})
        for (int j : {jx0, jx1})
            unknown.push_back(gid(i, j, ny));
    std::sort(unknown.begin(), unknown.end());
    unknown.erase(std::unique(unknown.begin(), unknown.end()), unknown.end());
    if (unknown.empty()) return;

    std::unordered_map<int, int> local_id;
    for (int p = 0; p < (int)unknown.size(); ++p) local_id[unknown[p]] = p;

    std::vector<Trip> triplets;
    for (int i = i_lo; i <= i_hi; ++i)
        for (int j = j_lo; j <= j_hi; ++j)
            append_godunov_row(triplets, u, m, n, i, j, ix0, jx0, ix1, jx1);

    SpMat G(nn, nn);
    G.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::SparseMatrix<double, Eigen::RowMajor> At = G.transpose();

    const int nloc = (int)unknown.size();
    std::vector<Trip> loc_trips;
    Eigen::VectorXd rhs(nloc);
    rhs.setZero();

    for (int p = 0; p < nloc; ++p) {
        const int row_g = unknown[p];
        rhs[p] = grad_u[row_g];
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(At, row_g); it; ++it) {
            const int col = (int)it.col();
            const double val = it.value();
            auto found = local_id.find(col);
            if (found != local_id.end())
                loc_trips.emplace_back(p, found->second, val);
            else
                rhs[p] -= val * res_out[col];
        }
    }

    SpMat A_loc(nloc, nloc);
    A_loc.setFromTriplets(loc_trips.begin(), loc_trips.end());
    Eigen::SparseLU<SpMat> solver;
    solver.analyzePattern(A_loc);
    solver.factorize(A_loc);
    if (solver.info() != Eigen::Success) {
        for (int id : unknown) res_out[id] = grad_u[id];
        return;
    }
    Eigen::VectorXd res_loc = solver.solve(rhs);
    for (int p = 0; p < nloc; ++p) res_out[unknown[p]] = res_loc[p];
}

static void backward(double *grad_f, const double *grad_u, const double *u, const double *f,
                     int m, int n, double h, double x, double y) {
    const int nx = m + 1, ny = n + 1, nn = nx * ny;
    const double area = h * h;
    const double lam_scale = 0.5;

    std::vector<double> delta(nn), lambda(nn), lam_scaled(nn), res(nn);
    for (int i = 0; i < nn; ++i) delta[i] = grad_u[i] / area;
    solve_adjoint_fsm_0f_src(lambda.data(), u, delta.data(), nx, ny, h, x, y, true);
    for (int i = 0; i < nn; ++i) lam_scaled[i] = lambda[i] * lam_scale;
    for (int i = 0; i < nn; ++i) grad_f[i] = lam_scaled[i] * 2.0 * f[i] * area;

    // Src correction: only the 4 source-box corners.
    patch_lu_corner_res_2d(res.data(), grad_u, u, lam_scaled.data(), m, n, x, y, 1);
    apply_source_simpson_grad_from_res(grad_f, res.data(), m, n, h, x, y);
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
    const bool use_source_fix = !(std::isnan(x) || std::isnan(y));
    auto lambda = torch::zeros_like(T);
    solve_adjoint_fsm_0f_src(lambda.data_ptr<double>(), T.data_ptr<double>(),
                             delta.data_ptr<double>(), nx, ny, h, x, y, use_source_fix);
    return lambda;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &eikonal_forward, "2D forward");
    m.def("backward", &eikonal_backward, "2D zero-flux FSM adjoint + src correction");
    m.def("solve_adjoint", &eikonal_solve_adjoint, "2D zero-flux FSM adjoint + src correction",
          pybind11::arg("T"), pybind11::arg("delta"), pybind11::arg("h"),
          pybind11::arg("x") = std::numeric_limits<double>::quiet_NaN(),
          pybind11::arg("y") = std::numeric_limits<double>::quiet_NaN());
}
