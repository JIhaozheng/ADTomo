// 3D continuous FSM adjoint.

#include <torch/extension.h>

#include <array>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

static inline int index3d(int i, int j, int k, int n, int l) {
    return i * n * l + j * l + k;
}

// Shared, ordered description of the eight corners of the source cell.
struct SourceCell {
    int ix0, jx0, kx0;
    int ix1, jx1, kx1;
    std::array<int, 8> ids;
    std::array<double, 8> weights;
    std::array<double, 8> distances;
};

static SourceCell make_source_cell(int m, int n, int l, double h, double x, double y, double z) {
    const int ix0 = std::max(0, std::min((int)std::floor(x), m - 1));
    const int jx0 = std::max(0, std::min((int)std::floor(y), n - 1));
    const int kx0 = std::max(0, std::min((int)std::floor(z), l - 1));
    const int ix1 = ix0 + 1;
    const int jx1 = jx0 + 1;
    const int kx1 = kx0 + 1;
    const double wx = x - ix0, wy = y - jx0, wz = z - kx0;
    const std::array<double, 8> weights = {
        (1 - wx) * (1 - wy) * (1 - wz), (1 - wx) * (1 - wy) * wz,
        (1 - wx) * wy * (1 - wz), (1 - wx) * wy * wz,
        wx * (1 - wy) * (1 - wz), wx * (1 - wy) * wz,
        wx * wy * (1 - wz), wx * wy * wz};
    const std::array<double, 8> distances = {
        std::sqrt((x - ix0) * (x - ix0) + (y - jx0) * (y - jx0) + (z - kx0) * (z - kx0)) * h,
        std::sqrt((x - ix0) * (x - ix0) + (y - jx0) * (y - jx0) + (z - kx1) * (z - kx1)) * h,
        std::sqrt((x - ix0) * (x - ix0) + (y - jx1) * (y - jx1) + (z - kx0) * (z - kx0)) * h,
        std::sqrt((x - ix0) * (x - ix0) + (y - jx1) * (y - jx1) + (z - kx1) * (z - kx1)) * h,
        std::sqrt((x - ix1) * (x - ix1) + (y - jx0) * (y - jx0) + (z - kx0) * (z - kx0)) * h,
        std::sqrt((x - ix1) * (x - ix1) + (y - jx0) * (y - jx0) + (z - kx1) * (z - kx1)) * h,
        std::sqrt((x - ix1) * (x - ix1) + (y - jx1) * (y - jx1) + (z - kx0) * (z - kx0)) * h,
        std::sqrt((x - ix1) * (x - ix1) + (y - jx1) * (y - jx1) + (z - kx1) * (z - kx1)) * h};
    return {ix0, jx0, kx0, ix1, jx1, kx1,
            {index3d(ix0, jx0, kx0, n, l), index3d(ix0, jx0, kx1, n, l),
             index3d(ix0, jx1, kx0, n, l), index3d(ix0, jx1, kx1, n, l),
             index3d(ix1, jx0, kx0, n, l), index3d(ix1, jx0, kx1, n, l),
             index3d(ix1, jx1, kx0, n, l), index3d(ix1, jx1, kx1, n, l)},
            weights, distances};
}

static double godunov_update(double a1_, double a2_, double a3_, double f, double h) {
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

static void forward_sweep(
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

                auto U = [&](int ii, int jj, int kk) { return u[index3d(ii, jj, kk, n, l)]; };
                auto F = [&](int ii, int jj, int kk) { return f[index3d(ii, jj, kk, n, l)]; };

                double uxmin = i == 0 ? U(i + 1, j, k)
                                      : (i == m - 1 ? U(i - 1, j, k)
                                                    : std::min(U(i + 1, j, k), U(i - 1, j, k)));
                double uymin = j == 0 ? U(i, j + 1, k)
                                      : (j == n - 1 ? U(i, j - 1, k)
                                                    : std::min(U(i, j + 1, k), U(i, j - 1, k)));
                double uzmin = k == 0 ? U(i, j, k + 1)
                                      : (k == l - 1 ? U(i, j, k - 1)
                                                    : std::min(U(i, j, k + 1), U(i, j, k - 1)));
                double u_new = godunov_update(uxmin, uymin, uzmin, F(i, j, k), h);
                u[index3d(i, j, k, n, l)] = std::min(u_new, u[index3d(i, j, k, n, l)]);
            }
}

static void sweep_all_directions(double *u, const double *f, int m, int n, int l, double h,
                     int ix0, int jx0, int kx0, int ix1, int jx1, int kx1) {
    forward_sweep(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, 1, 1, 1);
    forward_sweep(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, -1, 1, 1);
    forward_sweep(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, -1, -1, 1);
    forward_sweep(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, 1, -1, 1);
    forward_sweep(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, 1, -1, -1);
    forward_sweep(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, 1, 1, -1);
    forward_sweep(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, -1, 1, -1);
    forward_sweep(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, -1, -1, -1);
}

static void solve_forward(double *u, const double *f, double h,
                    int m, int n, int l, double x, double y, double z, double tol = 1e-8) {
    const auto source = make_source_cell(m, n, l, h, x, y, z);
    const int ix0 = source.ix0, jx0 = source.jx0, kx0 = source.kx0;
    const int ix1 = source.ix1, jx1 = source.jx1, kx1 = source.kx1;
    const int nn = m * n * l;

    for (int i = 0; i < nn; ++i) u[i] = 100000.0;

    auto F = [&](int ii, int jj, int kk) { return f[index3d(ii, jj, kk, n, l)]; };

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
        u[index3d(ii, jj, kk, n, l)] = (d / 6.0) * (fsrc + 4.0 * fmid + fcorner);
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
        sweep_all_directions(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1);
        double err = 0.0;
        for (int j = 0; j < nn; ++j) err = std::max(std::fabs(u[j] - u_old[j]), err);
        if (err < tol) break;
    }
}

static inline bool is_source_corner(int i, int j, int k,
                                    int ix0, int jx0, int kx0, int ix1, int jx1, int kx1) {
    return (i == ix0 || i == ix1) && (j == jx0 || j == jx1) && (k == kx0 || k == kx1);
}

static void upwind_split(double a, double &am, double &ap) {
    am = (a - std::fabs(a)) * 0.5;
    ap = (a + std::fabs(a)) * 0.5;
}

// Ordinary bulk stencil for div(lambda grad(-T)) = delta with zero exterior flux.
static double adjoint_update(const double *T, const double *lam, const double *delta,
                                   int m, int n, int l, int i, int j, int k, double h) {
    auto T_at = [&](int ii, int jj, int kk) -> double {
        if (ii < 0 || ii >= m || jj < 0 || jj >= n || kk < 0 || kk >= l) return 0.0;
        return T[index3d(ii, jj, kk, n, l)];
    };
    auto L_at = [&](int ii, int jj, int kk) -> double {
        if (ii < 0 || ii >= m || jj < 0 || jj >= n || kk < 0 || kk >= l) return 0.0;
        return lam[index3d(ii, jj, kk, n, l)];
    };

    double a1m = 0.0, a1p = 0.0, a2m = 0.0, a2p = 0.0;
    double b1m = 0.0, b1p = 0.0, b2m = 0.0, b2p = 0.0;
    double c1m = 0.0, c1p = 0.0, c2m = 0.0, c2p = 0.0;

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

    const double coe = (a2p - a1m) / h + (b2p - b1m) / h + (c2p - c1m) / h;
    if (std::fabs(coe) < 1e-15) return 0.0;

    const double hadj =
        ((a1p * L_at(i - 1, j, k) - a2m * L_at(i + 1, j, k)) / h +
         (b1p * L_at(i, j - 1, k) - b2m * L_at(i, j + 1, k)) / h +
         (c1p * L_at(i, j, k - 1) - c2m * L_at(i, j, k + 1)) / h);

    return (delta[index3d(i, j, k, n, l)] + hadj) / coe;
}

static void adjoint_sweep(double *lam, const double *T, const double *delta,
                                 int m, int n, int l, double h,
                                 int dirI, int dirJ, int dirK) {
    auto I = std::make_tuple(dirI == 1 ? 0 : m - 1, dirI == 1 ? m : -1, dirI);
    auto J = std::make_tuple(dirJ == 1 ? 0 : n - 1, dirJ == 1 ? n : -1, dirJ);
    auto K = std::make_tuple(dirK == 1 ? 0 : l - 1, dirK == 1 ? l : -1, dirK);

    for (int i = std::get<0>(I); i != std::get<1>(I); i += std::get<2>(I))
        for (int j = std::get<0>(J); j != std::get<1>(J); j += std::get<2>(J))
            for (int k = std::get<0>(K); k != std::get<1>(K); k += std::get<2>(K))
                lam[index3d(i, j, k, n, l)] =
                    adjoint_update(T, lam, delta, m, n, l, i, j, k, h);
}

static void solve_adjoint(double *lambda, const double *T, const double *delta,
                                     int m, int n, int l, double h,
                                     int max_iter = 200, double tol = 1e-6) {
    const int nn = m * n * l;
    std::fill(lambda, lambda + nn, 0.0);
    std::vector<double> lam_old(nn);

    for (int it = 0; it < max_iter; ++it) {
        lam_old.assign(lambda, lambda + nn);
        adjoint_sweep(lambda, T, delta, m, n, l, h, 1, 1, 1);
        adjoint_sweep(lambda, T, delta, m, n, l, h, -1, 1, 1);
        adjoint_sweep(lambda, T, delta, m, n, l, h, -1, -1, 1);
        adjoint_sweep(lambda, T, delta, m, n, l, h, 1, -1, 1);
        adjoint_sweep(lambda, T, delta, m, n, l, h, 1, -1, -1);
        adjoint_sweep(lambda, T, delta, m, n, l, h, 1, 1, -1);
        adjoint_sweep(lambda, T, delta, m, n, l, h, -1, 1, -1);
        adjoint_sweep(lambda, T, delta, m, n, l, h, -1, -1, -1);

        double err = 0.0;
        for (int idx = 0; idx < nn; ++idx)
            err = std::max(err, std::fabs(lambda[idx] - lam_old[idx]));
        if (err < tol) break;
    }
}

// Apply the exact transpose of the source-cell Simpson initialization.
static void source_gradient(
    double *grad_f, const SourceCell &source, const std::array<double, 8> &source_adjoint) {
    for (int parameter_corner = 0; parameter_corner < 8; ++parameter_corner) {
        double value = 0.0;
        for (int travel_time_corner = 0; travel_time_corner < 8; ++travel_time_corner) {
            value += source_adjoint[travel_time_corner] * source.distances[travel_time_corner] *
                     (source.weights[parameter_corner] + 0.5 +
                      (parameter_corner == travel_time_corner ? 1.0 : 0.0));
        }
        grad_f[source.ids[parameter_corner]] = value / 6.0;
    }
}

// The source-corner rows are pinned identities.  Their exact adjoint balance is
// grad_u at each corner minus contributions from neighboring ordinary rows.
static std::array<double, 8> source_adjoint(
    const SourceCell &source, const double *grad_u, const double *u, const double *lambda,
    int m, int n, int l, double lambda_scale) {
    const int ix0 = source.ix0, jx0 = source.jx0, kx0 = source.kx0;
    const int ix1 = source.ix1, jx1 = source.jx1, kx1 = source.kx1;

    std::array<double, 8> result;
    for (int corner = 0; corner < 8; ++corner) result[corner] = grad_u[source.ids[corner]];
    const auto corner_index = [&](int id) {
        for (int corner = 0; corner < 8; ++corner)
            if (source.ids[corner] == id) return corner;
        return -1;
    };

    for (int i = std::max(0, ix0 - 1); i <= std::min(m - 1, ix1 + 1); ++i) {
        for (int j = std::max(0, jx0 - 1); j <= std::min(n - 1, jx1 + 1); ++j) {
            for (int k = std::max(0, kx0 - 1); k <= std::min(l - 1, kx1 + 1); ++k) {
                if (is_source_corner(i, j, k, ix0, jx0, kx0, ix1, jx1, kx1)) continue;
                const int row = index3d(i, j, k, n, l);
                const auto U = [&](int ii, int jj, int kk) { return u[index3d(ii, jj, kk, n, l)]; };
                const double uxmin = i == 0 ? U(i + 1, j, k)
                    : (i == m - 1 ? U(i - 1, j, k) : std::min(U(i + 1, j, k), U(i - 1, j, k)));
                const double uymin = j == 0 ? U(i, j + 1, k)
                    : (j == n - 1 ? U(i, j - 1, k) : std::min(U(i, j + 1, k), U(i, j - 1, k)));
                const double uzmin = k == 0 ? U(i, j, k + 1)
                    : (k == l - 1 ? U(i, j, k - 1) : std::min(U(i, j, k + 1), U(i, j, k - 1)));
                const int idx = i == 0 ? index3d(i + 1, j, k, n, l)
                    : (i == m - 1 ? index3d(i - 1, j, k, n, l)
                    : (U(i + 1, j, k) > U(i - 1, j, k) ? index3d(i - 1, j, k, n, l) : index3d(i + 1, j, k, n, l)));
                const int idy = j == 0 ? index3d(i, j + 1, k, n, l)
                    : (j == n - 1 ? index3d(i, j - 1, k, n, l)
                    : (U(i, j + 1, k) > U(i, j - 1, k) ? index3d(i, j - 1, k, n, l) : index3d(i, j + 1, k, n, l)));
                const int idz = k == 0 ? index3d(i, j, k + 1, n, l)
                    : (k == l - 1 ? index3d(i, j, k - 1, n, l)
                    : (U(i, j, k + 1) > U(i, j, k - 1) ? index3d(i, j, k - 1, n, l) : index3d(i, j, k + 1, n, l)));
                const auto subtract = [&](int id, double coefficient) {
                    const int corner = corner_index(id);
                    if (corner >= 0) result[corner] -= coefficient * lambda[row] * lambda_scale;
                };
                if (U(i, j, k) > uxmin) subtract(idx, -2.0 * (U(i, j, k) - uxmin));
                if (U(i, j, k) > uymin) subtract(idy, -2.0 * (U(i, j, k) - uymin));
                if (U(i, j, k) > uzmin) subtract(idz, -2.0 * (U(i, j, k) - uzmin));
            }
        }
    }
    return result;
}

// Hybrid backward: FSM + src correction on source corners only.
static void solve_backward(
    double *grad_f, const double *grad_u, const double *u, const double *f, double h,
    int m, int n, int l, double x, double y, double z) {
    const int nn = m * n * l;
    const double vol = h * h * h;
    const double lam_scale = 0.5 * h;

    std::vector<double> delta(nn);
    for (int i = 0; i < nn; ++i) delta[i] = grad_u[i] / vol;

    std::vector<double> lambda(nn);
    solve_adjoint(lambda.data(), u, delta.data(), m, n, l, h);

    for (int i = 0; i < nn; ++i) grad_f[i] = lambda[i] * f[i] * vol;

    // Src correction: only the 8 source-box corners.
    const auto source = make_source_cell(m, n, l, h, x, y, z);
    const auto corner_adjoint = source_adjoint(source, grad_u, u, lambda.data(), m, n, l, lam_scale);
    source_gradient(grad_f, source, corner_adjoint);
}

// ---------------------------------------------------------------------------
// PyTorch bindings
// ---------------------------------------------------------------------------

torch::Tensor eikonal_forward(torch::Tensor f, double h, double x, double y, double z) {
    TORCH_CHECK(f.dim() == 3, "f must be a 3D tensor");
    TORCH_CHECK(f.is_contiguous(), "Input tensors must be contiguous");

    int m = f.size(0);
    int n = f.size(1);
    int l = f.size(2);

    auto u = torch::zeros_like(f);
    solve_forward(u.data_ptr<double>(), f.data_ptr<double>(), h, m, n, l, x, y, z);
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
    solve_backward(grad_f.data_ptr<double>(), grad_u.data_ptr<double>(), u.data_ptr<double>(),
             f.data_ptr<double>(), h, m, n, l, x, y, z);
    return grad_f;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &eikonal_forward, "3D forward");
    m.def("backward", &eikonal_backward, "3D backward");
}
