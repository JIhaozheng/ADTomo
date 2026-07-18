// 3D continuous Dirichlet FSM adjoint no source-corner (src) correction

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <tuple>
#include <vector>

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

static void upwind_split(double a, double &am, double &ap) {
    am = (a - std::fabs(a)) * 0.5;
    ap = (a + std::fabs(a)) * 0.5;
}

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

static double adjoint_stencil(const double *T, const double *lam, const double *delta,
                              int m, int n, int l, int i, int j, int k, double h,
                              double sx, double sy, double sz) {
    auto T_at = [&](int ii, int jj, int kk) -> double {
        if (ii < 0 || ii >= m || jj < 0 || jj >= n || kk < 0 || kk >= l) return 0.0;
        return T[gid(ii, jj, kk, n, l)];
    };
    auto L_at = [&](int ii, int jj, int kk) -> double {
        if (ii < 0 || ii >= m || jj < 0 || jj >= n || kk < 0 || kk >= l) return 0.0;
        return lam[gid(ii, jj, kk, n, l)];
    };

    if (i <= 0 || i >= m - 1 || j <= 0 || j >= n - 1 || k <= 0 || k >= l - 1) return 0.0;

    const int ix0 = std::max(0, std::min((int)std::floor(sx), m - 1));
    const int jx0 = std::max(0, std::min((int)std::floor(sy), n - 1));
    const int kx0 = std::max(0, std::min((int)std::floor(sz), l - 1));
    const int ix1 = ix0 + 1, jx1 = jx0 + 1, kx1 = kx0 + 1;

    double a1m = 0.0, a1p = 0.0, a2m = 0.0, a2p = 0.0;
    double b1m = 0.0, b1p = 0.0, b2m = 0.0, b2p = 0.0;
    double c1m = 0.0, c1p = 0.0, c2m = 0.0, c2p = 0.0;

    if (is_source_corner(i, j, k, ix0, jx0, kx0, ix1, jx1, kx1)) {
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
        const double a1 = -(T_at(i, j, k) - T_at(i - 1, j, k)) / h;
        const double a2 = -(T_at(i + 1, j, k) - T_at(i, j, k)) / h;
        const double b1 = -(T_at(i, j, k) - T_at(i, j - 1, k)) / h;
        const double b2 = -(T_at(i, j + 1, k) - T_at(i, j, k)) / h;
        const double c1 = -(T_at(i, j, k) - T_at(i, j, k - 1)) / h;
        const double c2 = -(T_at(i, j, k + 1) - T_at(i, j, k)) / h;
        upwind_split(a1, a1m, a1p);
        upwind_split(a2, a2m, a2p);
        upwind_split(b1, b1m, b1p);
        upwind_split(b2, b2m, b2p);
        upwind_split(c1, c1m, c1p);
        upwind_split(c2, c2m, c2p);
    }

    const double coe = (a2p - a1m) / h + (b2p - b1m) / h + (c2p - c1m) / h;
    if (std::fabs(coe) < 1e-15) return 0.0;

    const double hadj =
        ((a1p * L_at(i - 1, j, k) - a2m * L_at(i + 1, j, k)) / h +
         (b1p * L_at(i, j - 1, k) - b2m * L_at(i, j + 1, k)) / h +
         (c1p * L_at(i, j, k - 1) - c2m * L_at(i, j, k + 1)) / h);

    return (delta[gid(i, j, k, n, l)] + hadj) / coe;
}

static void zero_adjoint_boundaries(double *lam, int m, int n, int l) {
    for (int j = 0; j < n; ++j)
        for (int k = 0; k < l; ++k) {
            lam[gid(0, j, k, n, l)] = 0.0;
            lam[gid(m - 1, j, k, n, l)] = 0.0;
        }
    for (int i = 0; i < m; ++i)
        for (int k = 0; k < l; ++k) {
            lam[gid(i, 0, k, n, l)] = 0.0;
            lam[gid(i, n - 1, k, n, l)] = 0.0;
        }
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            lam[gid(i, j, 0, n, l)] = 0.0;
            lam[gid(i, j, l - 1, n, l)] = 0.0;
        }
}

static void sweep_adjoint(double *lam, const double *T, const double *delta,
                          int m, int n, int l, double h,
                          double sx, double sy, double sz,
                          int dirI, int dirJ, int dirK) {
    auto I = std::make_tuple(dirI == 1 ? 1 : m - 2, dirI == 1 ? m - 1 : 0, dirI);
    auto J = std::make_tuple(dirJ == 1 ? 1 : n - 2, dirJ == 1 ? n - 1 : 0, dirJ);
    auto K = std::make_tuple(dirK == 1 ? 1 : l - 2, dirK == 1 ? l - 1 : 0, dirK);

    for (int i = std::get<0>(I); i != std::get<1>(I); i += std::get<2>(I))
        for (int j = std::get<0>(J); j != std::get<1>(J); j += std::get<2>(J))
            for (int k = std::get<0>(K); k != std::get<1>(K); k += std::get<2>(K))
                lam[gid(i, j, k, n, l)] =
                    adjoint_stencil(T, lam, delta, m, n, l, i, j, k, h, sx, sy, sz);
}

static void solve_adjoint_fsm(double *lambda, const double *T, const double *delta,
                              int m, int n, int l, double h,
                              double sx, double sy, double sz,
                              int max_iter = 200, double tol = 1e-6) {
    const int nn = m * n * l;
    std::fill(lambda, lambda + nn, 0.0);
    std::vector<double> lam_old(nn);

    for (int it = 0; it < max_iter; ++it) {
        lam_old.assign(lambda, lambda + nn);
        sweep_adjoint(lambda, T, delta, m, n, l, h, sx, sy, sz, 1, 1, 1);
        sweep_adjoint(lambda, T, delta, m, n, l, h, sx, sy, sz, -1, 1, 1);
        sweep_adjoint(lambda, T, delta, m, n, l, h, sx, sy, sz, -1, -1, 1);
        sweep_adjoint(lambda, T, delta, m, n, l, h, sx, sy, sz, 1, -1, 1);
        sweep_adjoint(lambda, T, delta, m, n, l, h, sx, sy, sz, 1, -1, -1);
        sweep_adjoint(lambda, T, delta, m, n, l, h, sx, sy, sz, 1, 1, -1);
        sweep_adjoint(lambda, T, delta, m, n, l, h, sx, sy, sz, -1, 1, -1);
        sweep_adjoint(lambda, T, delta, m, n, l, h, sx, sy, sz, -1, -1, -1);
        zero_adjoint_boundaries(lambda, m, n, l);

        double err = 0.0;
        for (int idx = 0; idx < nn; ++idx)
            err = std::max(err, std::fabs(lambda[idx] - lam_old[idx]));
        if (err < tol) break;
    }
}

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
    solve_adjoint_fsm(lambda.data(), u, delta.data(), m, n, l, h, x, y, z);

    for (int i = 0; i < nn; ++i) grad_f[i] = (lambda[i] * lam_scale) * 2.0 * f[i] * h2;
}

// ---------------------------------------------------------------------------
// PyTorch interface — drop-in replacement for eikonal3d_op
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &eikonal_forward, "Eikonal3D forward (FSM sweep)");
    m.def("backward", &eikonal_backward,
          "Eikonal3D backward (Dirichlet FSM, nabla T, no src corner overwrite)");
}
