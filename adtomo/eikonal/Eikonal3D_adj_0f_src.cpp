// 3D continuous zero-flux FSM adjoint + source-corner (src) correction.
// Src (source corners only): local LU patch + Simpson overwrite (same as Eikonal3D.cpp).

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

// Layout matches Eikonal3D.cpp / eikonal3d_op:
//   f, u, grad_u : torch tensor shape (m, n, l), contiguous
//   x, y, z      : source in grid-index coordinates (not physical metres)
//   h            : grid spacing
//
// Default patch_radius=1 → only the 8 source-box corners are unknowns.

static inline int gid(int i, int j, int k, int n, int l) {
    return i * n * l + j * l + k;
}

static double calculate_unique_solution(double a1_, double a2_, double a3_, double f, double h) {
    double a1 = a1_, a2 = a2_, a3 = a3_;
    if (a1 > a2) std::swap(a1, a2);
    if (a1 > a3) std::swap(a1, a3);
    if (a2 > a3) std::swap(a2, a3);

    double x = a1 + f * h;
    if (x <= a2) return x;
    double B = -(a1 + a2);
    double C = (a1 * a1 + a2 * a2 - f * f * h * h) / 2.0;
    x = (-B + std::sqrt(B * B - 4.0 * C)) / 2.0;
    if (x <= a3) return x;
    B = -2.0 * (a1 + a2 + a3) / 3.0;
    C = (a1 * a1 + a2 * a2 + a3 * a3 - f * f * h * h) / 3.0;
    x = (-B + std::sqrt(B * B - 4.0 * C)) / 2.0;
    return x;
}

static void sweeping_over_I_J_K(
    double *u, const double *f, int m, int n, int l, double h,
    int ix0, int jx0, int kx0, int ix1, int jx1, int kx1,
    int dirI, int dirJ, int dirK) {
    auto I = std::make_tuple(dirI == 1 ? 0 : m - 1, dirI == 1 ? m : -1, dirI);
    auto J = std::make_tuple(dirJ == 1 ? 0 : n - 1, dirJ == 1 ? n : -1, dirJ);
    auto K = std::make_tuple(dirK == 1 ? 0 : l - 1, dirK == 1 ? l : -1, dirK);

    for (int i = std::get<0>(I); i != std::get<1>(I); i += std::get<2>(I))
        for (int j = std::get<0>(J); j != std::get<1>(J); j += std::get<2>(J))
            for (int k = std::get<0>(K); k != std::get<1>(K); k += std::get<2>(K)) {
                if ((i == ix0 && j == jx0 && k == kx0) || (i == ix0 && j == jx0 && k == kx1) ||
                    (i == ix0 && j == jx1 && k == kx0) || (i == ix0 && j == jx1 && k == kx1) ||
                    (i == ix1 && j == jx0 && k == kx0) || (i == ix1 && j == jx0 && k == kx1) ||
                    (i == ix1 && j == jx1 && k == kx0) || (i == ix1 && j == jx1 && k == kx1))
                    continue;

                auto U = [&](int ii, int jj, int kk) { return u[gid(ii, jj, kk, n, l)]; };
                auto F = [&](int ii, int jj, int kk) { return f[gid(ii, jj, kk, n, l)]; };

                double uxmin = i == 0 ? U(i + 1, j, k)
                                      : (i == m - 1 ? U(i - 1, j, k)
                                                    : std::min(U(i + 1, j, k), U(i - 1, j, k)));
                double uymin = j == 0 ? U(i, j + 1, k)
                                      : (j == n - 1 ? U(i, j - 1, k)
                                                    : std::min(U(i, j + 1, k), U(i, j - 1, k)));
                double uzmin = k == 0 ? U(i, j, k + 1)
                                      : (k == l - 1 ? U(i, j, k - 1)
                                                    : std::min(U(i, j, k + 1), U(i, j, k - 1)));
                double u_new = calculate_unique_solution(uxmin, uymin, uzmin, F(i, j, k), h);
                u[gid(i, j, k, n, l)] = std::min(u_new, u[gid(i, j, k, n, l)]);
            }
}

static void sweeping(double *u, const double *f, int m, int n, int l, double h,
                     int ix0, int jx0, int kx0, int ix1, int jx1, int kx1) {
    sweeping_over_I_J_K(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, 1, 1, 1);
    sweeping_over_I_J_K(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, -1, 1, 1);
    sweeping_over_I_J_K(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, -1, -1, 1);
    sweeping_over_I_J_K(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, 1, -1, 1);
    sweeping_over_I_J_K(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, 1, -1, -1);
    sweeping_over_I_J_K(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, 1, 1, -1);
    sweeping_over_I_J_K(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, -1, 1, -1);
    sweeping_over_I_J_K(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, -1, -1, -1);
}

static void forward(double *u, const double *f, double h,
                    int m, int n, int l, double x, double y, double z, double tol = 1e-8) {
    int ix0 = std::max(0, std::min((int)std::floor(x), m - 1));
    int jx0 = std::max(0, std::min((int)std::floor(y), n - 1));
    int kx0 = std::max(0, std::min((int)std::floor(z), l - 1));
    int ix1 = ix0 + 1;
    int jx1 = jx0 + 1;
    int kx1 = kx0 + 1;
    const int nn = m * n * l;

    for (int i = 0; i < nn; ++i) u[i] = 100000.0;

    auto F = [&](int ii, int jj, int kk) { return f[gid(ii, jj, kk, n, l)]; };

    double f000 = F(ix0, jx0, kx0);
    double f001 = F(ix0, jx0, kx1);
    double f010 = F(ix0, jx1, kx0);
    double f011 = F(ix0, jx1, kx1);
    double f100 = F(ix1, jx0, kx0);
    double f101 = F(ix1, jx0, kx1);
    double f110 = F(ix1, jx1, kx0);
    double f111 = F(ix1, jx1, kx1);

    double wx = x - ix0;
    double wy = y - jx0;
    double wz = z - kx0;
    double fsrc = (1 - wx) * (1 - wy) * (1 - wz) * f000 + (1 - wx) * (1 - wy) * wz * f001 +
                  (1 - wx) * wy * (1 - wz) * f010 + (1 - wx) * wy * wz * f011 +
                  wx * (1 - wy) * (1 - wz) * f100 + wx * (1 - wy) * wz * f101 +
                  wx * wy * (1 - wz) * f110 + wx * wy * wz * f111;
    double fmid = (f000 + f001 + f010 + f011 + f100 + f101 + f110 + f111) / 8.0;

    auto set_corner = [&](int ii, int jj, int kk, double fcorner) {
        double d = std::sqrt((x - ii) * (x - ii) + (y - jj) * (y - jj) + (z - kk) * (z - kk)) * h;
        u[gid(ii, jj, kk, n, l)] = (d / 6.0) * (fsrc + 4.0 * fmid + fcorner);
    };

    set_corner(ix0, jx0, kx0, f000);
    set_corner(ix0, jx0, kx1, f001);
    set_corner(ix0, jx1, kx0, f010);
    set_corner(ix0, jx1, kx1, f011);
    set_corner(ix1, jx0, kx0, f100);
    set_corner(ix1, jx0, kx1, f101);
    set_corner(ix1, jx1, kx0, f110);
    set_corner(ix1, jx1, kx1, f111);

    std::vector<double> u_old(nn);
    for (int it = 0; it < 20; ++it) {
        u_old.assign(u, u + nn);
        sweeping(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1);
        double err = 0.0;
        for (int j = 0; j < nn; ++j) err = std::max(std::fabs(u[j] - u_old[j]), err);
        if (err < tol) break;
    }
}

static inline bool is_source_corner(int i, int j, int k,
                                    int ix0, int jx0, int kx0, int ix1, int jx1, int kx1) {
    return (i == ix0 || i == ix1) && (j == jx0 || j == jx1) && (k == kx0 || k == kx1);
}

// Normal-projected ∇(-T) toward source at source-box nodes (EikonalSolvers adjderivonsource).
template <typename TAt>
static void adjderivonsource(
    const TAt &T_at, int m, int n, int l, int i, int j, int k, double h,
    double sx, double sy, double sz,
    int ix0, int jx0, int kx0, int ix1, int jx1, int kx1,
    double &aback, double &aforw, double &bback, double &bforw, double &cback, double &cforw) {
    const double Ti = T_at(i, j, k);
    const double distPSx = std::fabs(sx - static_cast<double>(i)) * h;
    const double distPSy = std::fabs(sy - static_cast<double>(j)) * h;
    const double distPSz = std::fabs(sz - static_cast<double>(k)) * h;
    const double dist2src = std::hypot(sx - i, sy - j, sz - k) * h;

    aback = aforw = bback = bforw = cback = cforw = 0.0;
    if (dist2src < 1e-15) return;

    const double thx = Ti * std::hypot(distPSy, distPSz) / dist2src;
    const double thy = Ti * std::hypot(distPSx, distPSz) / dist2src;
    const double thz = Ti * std::hypot(distPSx, distPSy) / dist2src;

    auto is_sc = [&](int ii, int jj, int kk) {
        return is_source_corner(ii, jj, kk, ix0, jx0, kx0, ix1, jx1, kx1);
    };

    if (i < m - 1 && is_sc(i + 1, j, k)) {
        if (i > 0) aback = -(Ti - T_at(i - 1, j, k)) / h;
        aforw = (distPSx < 1e-15) ? 0.0 : -(thx - Ti) / distPSx;
    } else if (i > 0 && is_sc(i - 1, j, k)) {
        aback = (distPSx < 1e-15) ? 0.0 : -(Ti - thx) / distPSx;
        if (i < m - 1) aforw = -(T_at(i + 1, j, k) - Ti) / h;
    } else {
        if (i > 0) aback = -(Ti - T_at(i - 1, j, k)) / h;
        if (i < m - 1) aforw = -(T_at(i + 1, j, k) - Ti) / h;
    }

    if (j < n - 1 && is_sc(i, j + 1, k)) {
        if (j > 0) bback = -(Ti - T_at(i, j - 1, k)) / h;
        bforw = (distPSy < 1e-15) ? 0.0 : -(thy - Ti) / distPSy;
    } else if (j > 0 && is_sc(i, j - 1, k)) {
        bback = (distPSy < 1e-15) ? 0.0 : -(Ti - thy) / distPSy;
        if (j < n - 1) bforw = -(T_at(i, j + 1, k) - Ti) / h;
    } else {
        if (j > 0) bback = -(Ti - T_at(i, j - 1, k)) / h;
        if (j < n - 1) bforw = -(T_at(i, j + 1, k) - Ti) / h;
    }

    if (k < l - 1 && is_sc(i, j, k + 1)) {
        if (k > 0) cback = -(Ti - T_at(i, j, k - 1)) / h;
        cforw = (distPSz < 1e-15) ? 0.0 : -(thz - Ti) / distPSz;
    } else if (k > 0 && is_sc(i, j, k - 1)) {
        cback = (distPSz < 1e-15) ? 0.0 : -(Ti - thz) / distPSz;
        if (k < l - 1) cforw = -(T_at(i, j, k + 1) - Ti) / h;
    } else {
        if (k > 0) cback = -(Ti - T_at(i, j, k - 1)) / h;
        if (k < l - 1) cforw = -(T_at(i, j, k + 1) - Ti) / h;
    }
}

static void upwind_split(double a, double &am, double &ap) {
    am = (a - std::fabs(a)) * 0.5;
    ap = (a + std::fabs(a)) * 0.5;
}

// Upwind stencil for nabla·(lambda nabla(-T)) = delta with zero exterior flux.
// At source-box corners: use adjderivonsource for ∇(-T); elsewhere unchanged.
static double adjoint_stencil_0f_src(const double *T, const double *lam, const double *delta,
                                     int m, int n, int l, int i, int j, int k, double h,
                                     double sx, double sy, double sz, bool use_source_fix) {
    auto T_at = [&](int ii, int jj, int kk) -> double {
        if (ii < 0 || ii >= m || jj < 0 || jj >= n || kk < 0 || kk >= l) return 0.0;
        return T[gid(ii, jj, kk, n, l)];
    };
    auto L_at = [&](int ii, int jj, int kk) -> double {
        if (ii < 0 || ii >= m || jj < 0 || jj >= n || kk < 0 || kk >= l) return 0.0;
        return lam[gid(ii, jj, kk, n, l)];
    };

    const int ix0 = std::max(0, std::min((int)std::floor(sx), m - 1));
    const int jx0 = std::max(0, std::min((int)std::floor(sy), n - 1));
    const int kx0 = std::max(0, std::min((int)std::floor(sz), l - 1));
    const int ix1 = ix0 + 1;
    const int jx1 = jx0 + 1;
    const int kx1 = kx0 + 1;

    double a1m = 0.0, a1p = 0.0, a2m = 0.0, a2p = 0.0;
    double b1m = 0.0, b1p = 0.0, b2m = 0.0, b2p = 0.0;
    double c1m = 0.0, c1p = 0.0, c2m = 0.0, c2p = 0.0;

    if (use_source_fix && is_source_corner(i, j, k, ix0, jx0, kx0, ix1, jx1, kx1)) {
        double aback, aforw, bback, bforw, cback, cforw;
        adjderivonsource(T_at, m, n, l, i, j, k, h, sx, sy, sz,
                         ix0, jx0, kx0, ix1, jx1, kx1, aback, aforw, bback, bforw, cback, cforw);
        upwind_split(aback, a1m, a1p);
        upwind_split(aforw, a2m, a2p);
        upwind_split(bback, b1m, b1p);
        upwind_split(bforw, b2m, b2p);
        upwind_split(cback, c1m, c1p);
        upwind_split(cforw, c2m, c2p);
    } else {
        const bool at_west = (i == 0);
        const bool at_east = (i == m - 1);
        const bool at_south = (j == 0);
        const bool at_north = (j == n - 1);
        const bool at_bottom = (k == 0);
        const bool at_top = (k == l - 1);

        if (!at_west) {
            const double a1 = -(T_at(i, j, k) - T_at(i - 1, j, k)) / h;
            upwind_split(a1, a1m, a1p);
        }
        if (!at_east) {
            const double a2 = -(T_at(i + 1, j, k) - T_at(i, j, k)) / h;
            upwind_split(a2, a2m, a2p);
        }
        if (!at_south) {
            const double b1 = -(T_at(i, j, k) - T_at(i, j - 1, k)) / h;
            upwind_split(b1, b1m, b1p);
        }
        if (!at_north) {
            const double b2 = -(T_at(i, j + 1, k) - T_at(i, j, k)) / h;
            upwind_split(b2, b2m, b2p);
        }
        if (!at_bottom) {
            const double c1 = -(T_at(i, j, k) - T_at(i, j, k - 1)) / h;
            upwind_split(c1, c1m, c1p);
        }
        if (!at_top) {
            const double c2 = -(T_at(i, j, k + 1) - T_at(i, j, k)) / h;
            upwind_split(c2, c2m, c2p);
        }
    }

    const double coe = (a2p - a1m) / h + (b2p - b1m) / h + (c2p - c1m) / h;
    if (std::fabs(coe) < 1e-15) return 0.0;

    const double hadj =
        ((a1p * L_at(i - 1, j, k) - a2m * L_at(i + 1, j, k)) / h +
         (b1p * L_at(i, j - 1, k) - b2m * L_at(i, j + 1, k)) / h +
         (c1p * L_at(i, j, k - 1) - c2m * L_at(i, j, k + 1)) / h);

    return (delta[gid(i, j, k, n, l)] + hadj) / coe;
}

static void sweep_adjoint_0f_src(double *lam, const double *T, const double *delta,
                                 int m, int n, int l, double h,
                                 double sx, double sy, double sz, bool use_source_fix,
                                 int dirI, int dirJ, int dirK) {
    auto I = std::make_tuple(dirI == 1 ? 0 : m - 1, dirI == 1 ? m : -1, dirI);
    auto J = std::make_tuple(dirJ == 1 ? 0 : n - 1, dirJ == 1 ? n : -1, dirJ);
    auto K = std::make_tuple(dirK == 1 ? 0 : l - 1, dirK == 1 ? l : -1, dirK);

    for (int i = std::get<0>(I); i != std::get<1>(I); i += std::get<2>(I))
        for (int j = std::get<0>(J); j != std::get<1>(J); j += std::get<2>(J))
            for (int k = std::get<0>(K); k != std::get<1>(K); k += std::get<2>(K))
                lam[gid(i, j, k, n, l)] =
                    adjoint_stencil_0f_src(T, lam, delta, m, n, l, i, j, k, h,
                                           sx, sy, sz, use_source_fix);
}

static void solve_adjoint_fsm_0f_src(double *lambda, const double *T, const double *delta,
                                     int m, int n, int l, double h,
                                     double sx, double sy, double sz, bool use_source_fix,
                                     int max_iter = 200, double tol = 1e-6) {
    const int nn = m * n * l;
    std::fill(lambda, lambda + nn, 0.0);
    std::vector<double> lam_old(nn);

    for (int it = 0; it < max_iter; ++it) {
        lam_old.assign(lambda, lambda + nn);
        sweep_adjoint_0f_src(lambda, T, delta, m, n, l, h, sx, sy, sz, use_source_fix, 1, 1, 1);
        sweep_adjoint_0f_src(lambda, T, delta, m, n, l, h, sx, sy, sz, use_source_fix, -1, 1, 1);
        sweep_adjoint_0f_src(lambda, T, delta, m, n, l, h, sx, sy, sz, use_source_fix, -1, -1, 1);
        sweep_adjoint_0f_src(lambda, T, delta, m, n, l, h, sx, sy, sz, use_source_fix, 1, -1, 1);
        sweep_adjoint_0f_src(lambda, T, delta, m, n, l, h, sx, sy, sz, use_source_fix, 1, -1, -1);
        sweep_adjoint_0f_src(lambda, T, delta, m, n, l, h, sx, sy, sz, use_source_fix, 1, 1, -1);
        sweep_adjoint_0f_src(lambda, T, delta, m, n, l, h, sx, sy, sz, use_source_fix, -1, 1, -1);
        sweep_adjoint_0f_src(lambda, T, delta, m, n, l, h, sx, sy, sz, use_source_fix, -1, -1, -1);

        double err = 0.0;
        for (int idx = 0; idx < nn; ++idx)
            err = std::max(err, std::fabs(lambda[idx] - lam_old[idx]));
        if (err < tol) break;
    }
}

// Simpson corner overwrite using discrete patch residual (same formula as Eikonal3D.cpp).
static void apply_source_simpson_grad_from_res(
    double *grad_f, const double *res,
    int m, int n, int l, double h, double x, double y, double z) {
    int ix0 = std::max(0, std::min((int)std::floor(x), m - 1));
    int jx0 = std::max(0, std::min((int)std::floor(y), n - 1));
    int kx0 = std::max(0, std::min((int)std::floor(z), l - 1));
    int ix1 = ix0 + 1;
    int jx1 = jx0 + 1;
    int kx1 = kx0 + 1;

    double wx = x - ix0;
    double wy = y - jx0;
    double wz = z - kx0;
    double w000 = (1 - wx) * (1 - wy) * (1 - wz);
    double w001 = (1 - wx) * (1 - wy) * wz;
    double w010 = (1 - wx) * wy * (1 - wz);
    double w011 = (1 - wx) * wy * wz;
    double w100 = wx * (1 - wy) * (1 - wz);
    double w101 = wx * (1 - wy) * wz;
    double w110 = wx * wy * (1 - wz);
    double w111 = wx * wy * wz;

    double res000 = res[gid(ix0, jx0, kx0, n, l)];
    double res001 = res[gid(ix0, jx0, kx1, n, l)];
    double res010 = res[gid(ix0, jx1, kx0, n, l)];
    double res011 = res[gid(ix0, jx1, kx1, n, l)];
    double res100 = res[gid(ix1, jx0, kx0, n, l)];
    double res101 = res[gid(ix1, jx0, kx1, n, l)];
    double res110 = res[gid(ix1, jx1, kx0, n, l)];
    double res111 = res[gid(ix1, jx1, kx1, n, l)];

    auto dist = [&](int ii, int jj, int kk) {
        return std::sqrt((x - ii) * (x - ii) + (y - jj) * (y - jj) + (z - kk) * (z - kk)) * h;
    };
    double d000 = dist(ix0, jx0, kx0);
    double d001 = dist(ix0, jx0, kx1);
    double d010 = dist(ix0, jx1, kx0);
    double d011 = dist(ix0, jx1, kx1);
    double d100 = dist(ix1, jx0, kx0);
    double d101 = dist(ix1, jx0, kx1);
    double d110 = dist(ix1, jx1, kx0);
    double d111 = dist(ix1, jx1, kx1);

    grad_f[gid(ix0, jx0, kx0, n, l)] =
        (res000 * d000 * (w000 + 0.5 + 1) + res001 * d001 * (w000 + 0.5) + res010 * d010 * (w000 + 0.5) +
         res011 * d011 * (w000 + 0.5) + res100 * d100 * (w000 + 0.5) + res101 * d101 * (w000 + 0.5) +
         res110 * d110 * (w000 + 0.5) + res111 * d111 * (w000 + 0.5)) /
        6.0;
    grad_f[gid(ix0, jx0, kx1, n, l)] =
        (res000 * d000 * (w001 + 0.5) + res001 * d001 * (w001 + 0.5 + 1) + res010 * d010 * (w001 + 0.5) +
         res011 * d011 * (w001 + 0.5) + res100 * d100 * (w001 + 0.5) + res101 * d101 * (w001 + 0.5) +
         res110 * d110 * (w001 + 0.5) + res111 * d111 * (w001 + 0.5)) /
        6.0;
    grad_f[gid(ix0, jx1, kx0, n, l)] =
        (res000 * d000 * (w010 + 0.5) + res001 * d001 * (w010 + 0.5) + res010 * d010 * (w010 + 0.5 + 1) +
         res011 * d011 * (w010 + 0.5) + res100 * d100 * (w010 + 0.5) + res101 * d101 * (w010 + 0.5) +
         res110 * d110 * (w010 + 0.5) + res111 * d111 * (w010 + 0.5)) /
        6.0;
    grad_f[gid(ix0, jx1, kx1, n, l)] =
        (res000 * d000 * (w011 + 0.5) + res001 * d001 * (w011 + 0.5) + res010 * d010 * (w011 + 0.5) +
         res011 * d011 * (w011 + 0.5 + 1) + res100 * d100 * (w011 + 0.5) + res101 * d101 * (w011 + 0.5) +
         res110 * d110 * (w011 + 0.5) + res111 * d111 * (w011 + 0.5)) /
        6.0;
    grad_f[gid(ix1, jx0, kx0, n, l)] =
        (res000 * d000 * (w100 + 0.5) + res001 * d001 * (w100 + 0.5) + res010 * d010 * (w100 + 0.5) +
         res011 * d011 * (w100 + 0.5) + res100 * d100 * (w100 + 0.5 + 1) + res101 * d101 * (w100 + 0.5) +
         res110 * d110 * (w100 + 0.5) + res111 * d111 * (w100 + 0.5)) /
        6.0;
    grad_f[gid(ix1, jx0, kx1, n, l)] =
        (res000 * d000 * (w101 + 0.5) + res001 * d001 * (w101 + 0.5) + res010 * d010 * (w101 + 0.5) +
         res011 * d011 * (w101 + 0.5) + res100 * d100 * (w101 + 0.5) + res101 * d101 * (w101 + 0.5 + 1) +
         res110 * d110 * (w101 + 0.5) + res111 * d111 * (w101 + 0.5)) /
        6.0;
    grad_f[gid(ix1, jx1, kx0, n, l)] =
        (res000 * d000 * (w110 + 0.5) + res001 * d001 * (w110 + 0.5) + res010 * d010 * (w110 + 0.5) +
         res011 * d011 * (w110 + 0.5) + res100 * d100 * (w110 + 0.5) + res101 * d101 * (w110 + 0.5) +
         res110 * d110 * (w110 + 0.5 + 1) + res111 * d111 * (w110 + 0.5)) /
        6.0;
    grad_f[gid(ix1, jx1, kx1, n, l)] =
        (res000 * d000 * (w111 + 0.5) + res001 * d001 * (w111 + 0.5) + res010 * d010 * (w111 + 0.5) +
         res011 * d011 * (w111 + 0.5) + res100 * d100 * (w111 + 0.5) + res101 * d101 * (w111 + 0.5) +
         res110 * d110 * (w111 + 0.5) + res111 * d111 * (w111 + 0.5 + 1)) /
        6.0;
}

// Assemble one Godunov Jacobian row for node (i,j,k) into triplets (same as Eikonal3D.cpp).
static void append_godunov_row(
    std::vector<Trip> &triplets, const double *u, int m, int n, int l,
    int i, int j, int k, int ix0, int jx0, int kx0, int ix1, int jx1, int kx1) {
    int this_id = gid(i, j, k, n, l);
    if (is_source_corner(i, j, k, ix0, jx0, kx0, ix1, jx1, kx1)) {
        triplets.emplace_back(this_id, this_id, 1.0);
        return;
    }

    auto U = [&](int ii, int jj, int kk) { return u[gid(ii, jj, kk, n, l)]; };

    double uxmin = i == 0 ? U(i + 1, j, k)
                          : (i == m - 1 ? U(i - 1, j, k) : std::min(U(i + 1, j, k), U(i - 1, j, k)));
    double uymin = j == 0 ? U(i, j + 1, k)
                          : (j == n - 1 ? U(i, j - 1, k) : std::min(U(i, j + 1, k), U(i, j - 1, k)));
    double uzmin = k == 0 ? U(i, j, k + 1)
                          : (k == l - 1 ? U(i, j, k - 1) : std::min(U(i, j, k + 1), U(i, j, k - 1)));

    int idx = i == 0 ? gid(i + 1, j, k, n, l)
                     : (i == m - 1 ? gid(i - 1, j, k, n, l)
                                   : (U(i + 1, j, k) > U(i - 1, j, k) ? gid(i - 1, j, k, n, l)
                                                                      : gid(i + 1, j, k, n, l)));
    int idy = j == 0 ? gid(i, j + 1, k, n, l)
                     : (j == n - 1 ? gid(i, j - 1, k, n, l)
                                   : (U(i, j + 1, k) > U(i, j - 1, k) ? gid(i, j - 1, k, n, l)
                                                                      : gid(i, j + 1, k, n, l)));
    int idz = k == 0 ? gid(i, j, k + 1, n, l)
                     : (k == l - 1 ? gid(i, j, k - 1, n, l)
                                   : (U(i, j, k + 1) > U(i, j, k - 1) ? gid(i, j, k - 1, n, l)
                                                                      : gid(i, j, k + 1, n, l)));

    if (U(i, j, k) > uxmin) {
        triplets.emplace_back(this_id, this_id, 2.0 * (U(i, j, k) - uxmin));
        triplets.emplace_back(this_id, idx, -2.0 * (U(i, j, k) - uxmin));
    }
    if (U(i, j, k) > uymin) {
        triplets.emplace_back(this_id, this_id, 2.0 * (U(i, j, k) - uymin));
        triplets.emplace_back(this_id, idy, -2.0 * (U(i, j, k) - uymin));
    }
    if (U(i, j, k) > uzmin) {
        triplets.emplace_back(this_id, this_id, 2.0 * (U(i, j, k) - uzmin));
        triplets.emplace_back(this_id, idz, -2.0 * (U(i, j, k) - uzmin));
    }
}

// Local discrete adjoint patch (3D analog of lu_patch_corner.py).
// Unknowns: the 8 source-box corners (works on domain boundary).
// Other nodes in the radius-enlarged patch: Dirichlet res = lam_scaled.
static void patch_lu_corner_res_3d(
    double *res_out, const double *grad_u, const double *u, const double *lam_scaled,
    int m, int n, int l, double x, double y, double z, int radius = 1) {
    const int nn = m * n * l;
    int ix0 = std::max(0, std::min((int)std::floor(x), m - 1));
    int jx0 = std::max(0, std::min((int)std::floor(y), n - 1));
    int kx0 = std::max(0, std::min((int)std::floor(z), l - 1));
    int ix1 = std::min(ix0 + 1, m - 1);
    int jx1 = std::min(jx0 + 1, n - 1);
    int kx1 = std::min(kx0 + 1, l - 1);

    int i_lo = std::max(0, ix0 - radius);
    int i_hi = std::min(m - 1, ix1 + radius);
    int j_lo = std::max(0, jx0 - radius);
    int j_hi = std::min(n - 1, jx1 + radius);
    int k_lo = std::max(0, kx0 - radius);
    int k_hi = std::min(l - 1, kx1 + radius);

    std::memcpy(res_out, lam_scaled, sizeof(double) * nn);

    // Unknowns = source corners (may be <8 if source sits on a domain face).
    std::vector<int> unknown;
    unknown.reserve(8);
    for (int i : {ix0, ix1})
        for (int j : {jx0, jx1})
            for (int k : {kx0, kx1})
                unknown.push_back(gid(i, j, k, n, l));
    // Unique (in case ix0==ix1 etc.)
    std::sort(unknown.begin(), unknown.end());
    unknown.erase(std::unique(unknown.begin(), unknown.end()), unknown.end());
    if (unknown.empty()) return;

    std::unordered_map<int, int> local_id;
    local_id.reserve(unknown.size() * 2);
    for (int p = 0; p < (int)unknown.size(); ++p) local_id[unknown[p]] = p;

    std::vector<Trip> triplets;
    triplets.reserve((i_hi - i_lo + 1) * (j_hi - j_lo + 1) * (k_hi - k_lo + 1) * 6);
    for (int i = i_lo; i <= i_hi; ++i)
        for (int j = j_lo; j <= j_hi; ++j)
            for (int k = k_lo; k <= k_hi; ++k)
                append_godunov_row(triplets, u, m, n, l, i, j, k, ix0, jx0, kx0, ix1, jx1, kx1);

    SpMat G(nn, nn);
    G.setFromTriplets(triplets.begin(), triplets.end());
    // Row-major so we can iterate equations (rows of Gᵀ) for each unknown.
    Eigen::SparseMatrix<double, Eigen::RowMajor> At = G.transpose();

    const int nloc = (int)unknown.size();
    std::vector<Trip> loc_trips;
    loc_trips.reserve(nloc * 16);
    Eigen::VectorXd rhs(nloc);
    rhs.setZero();

    for (int p = 0; p < nloc; ++p) {
        const int row_g = unknown[p];
        rhs[p] = grad_u[row_g];
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(At, row_g); it; ++it) {
            const int col = (int)it.col();
            const double val = it.value();
            auto found = local_id.find(col);
            if (found != local_id.end()) {
                loc_trips.emplace_back(p, found->second, val);
            } else {
                rhs[p] -= val * res_out[col];
            }
        }
    }

    SpMat A_loc(nloc, nloc);
    A_loc.setFromTriplets(loc_trips.begin(), loc_trips.end());
    Eigen::SparseLU<SpMat> solver;
    solver.analyzePattern(A_loc);
    solver.factorize(A_loc);
    if (solver.info() != Eigen::Success) {
        // Fall back: pinned corners → res = grad_u (identity rows only).
        for (int id : unknown) res_out[id] = grad_u[id];
        return;
    }
    Eigen::VectorXd res_loc = solver.solve(rhs);
    for (int p = 0; p < nloc; ++p) res_out[unknown[p]] = res_loc[p];
}

// Hybrid backward: FSM bulk + src correction on source corners only.
static void backward(
    double *grad_f, const double *grad_u, const double *u, const double *f, double h,
    int m, int n, int l, double x, double y, double z) {
    const int nn = m * n * l;
    const double vol = h * h * h;
    const double h2 = h * h;
    const double lam_scale = 0.5 * h;

    std::vector<double> delta(nn);
    for (int i = 0; i < nn; ++i) delta[i] = grad_u[i] / vol;

    std::vector<double> lambda(nn);
    solve_adjoint_fsm_0f_src(lambda.data(), u, delta.data(), m, n, l, h, x, y, z, true);

    std::vector<double> lam_scaled(nn);
    for (int i = 0; i < nn; ++i) lam_scaled[i] = lambda[i] * lam_scale;

    for (int i = 0; i < nn; ++i) grad_f[i] = lam_scaled[i] * 2.0 * f[i] * h2;

    // Src correction: only the 8 source-box corners.
    std::vector<double> res(nn);
    patch_lu_corner_res_3d(res.data(), grad_u, u, lam_scaled.data(), m, n, l, x, y, z, 1);
    apply_source_simpson_grad_from_res(grad_f, res.data(), m, n, l, h, x, y, z);
}

// ---------------------------------------------------------------------------
// PyTorch interface — drop-in replacement for eikonal3d_adj_0f_op
// ---------------------------------------------------------------------------

torch::Tensor eikonal_forward(torch::Tensor f, double h, double x, double y, double z) {
    TORCH_CHECK(f.dim() == 3, "f must be a 3D tensor");
    TORCH_CHECK(f.is_contiguous(), "Input tensors must be contiguous");

    int m = f.size(0);
    int n = f.size(1);
    int l = f.size(2);

    auto u = torch::zeros_like(f);
    forward(u.data_ptr<double>(), f.data_ptr<double>(), h, m, n, l, x, y, z);
    return u;
}

torch::Tensor eikonal_backward(torch::Tensor grad_u, torch::Tensor u, torch::Tensor f,
                               double h, double x, double y, double z) {
    TORCH_CHECK(grad_u.dim() == 3 && u.dim() == 3 && f.dim() == 3, "All tensors must be 3D");
    TORCH_CHECK(grad_u.sizes() == u.sizes(), "All tensors must have the same size");
    TORCH_CHECK(grad_u.is_contiguous() && u.is_contiguous() && f.is_contiguous(),
                "All tensors must be contiguous");

    int m = u.size(0);
    int n = u.size(1);
    int l = u.size(2);

    auto grad_f = torch::zeros_like(f);
    backward(grad_f.data_ptr<double>(), grad_u.data_ptr<double>(), u.data_ptr<double>(),
             f.data_ptr<double>(), h, m, n, l, x, y, z);
    return grad_f;
}

torch::Tensor eikonal_solve_adjoint(torch::Tensor T, torch::Tensor delta, double h,
                                     double x, double y, double z) {
    TORCH_CHECK(T.dim() == 3 && delta.dim() == 3, "T and delta must be 3D tensors");
    TORCH_CHECK(T.sizes() == delta.sizes(), "T and delta must have the same shape");
    TORCH_CHECK(T.is_contiguous() && delta.is_contiguous(), "T and delta must be contiguous");

    int m = T.size(0);
    int n = T.size(1);
    int l = T.size(2);

    const bool use_source_fix = !(std::isnan(x) || std::isnan(y) || std::isnan(z));

    auto lambda = torch::zeros_like(T);
    solve_adjoint_fsm_0f_src(lambda.data_ptr<double>(), T.data_ptr<double>(),
                             delta.data_ptr<double>(), m, n, l, h, x, y, z, use_source_fix);
    return lambda;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &eikonal_forward, "Eikonal3D forward (FSM sweep)");
    m.def("backward", &eikonal_backward,
          "Eikonal3D backward (FSM + src corner LU/Simpson overwrite)");
    m.def("solve_adjoint", &eikonal_solve_adjoint,
          "Eikonal3D adjoint solve with optional source coords (NaN => no source-box fix)",
          pybind11::arg("T"), pybind11::arg("delta"), pybind11::arg("h"),
          pybind11::arg("x") = std::numeric_limits<double>::quiet_NaN(),
          pybind11::arg("y") = std::numeric_limits<double>::quiet_NaN(),
          pybind11::arg("z") = std::numeric_limits<double>::quiet_NaN());
}
