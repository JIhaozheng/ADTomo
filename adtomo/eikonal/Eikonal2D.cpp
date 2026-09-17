// 2D continuous FSM adjoint.

#include <torch/extension.h>

#include <array>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <tuple>
#include <unordered_map>
#include <vector>

static constexpr double FSM_INF = 1.0e20;

static inline int gid(int i, int j, int ny) { return i * ny + j; }

// Shared, ordered description of the four corners of the source cell.
struct SourceCell2D {
    int ix0, iy0, ix1, iy1;
    std::array<int, 4> ids;
    double wx, wy;
    std::array<double, 4> weights;
    std::array<double, 4> distances;
};

static SourceCell2D make_source_cell_2d(int m, int n, double h, double x, double y) {
    const int ny = n + 1;
    const int ix0 = std::max(0, std::min((int)std::floor(x), m - 1));
    const int iy0 = std::max(0, std::min((int)std::floor(y), n - 1));
    const int ix1 = ix0 + 1, iy1 = iy0 + 1;
    const double wx = x - ix0, wy = y - iy0;
    const std::array<double, 4> weights = {
        (1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy};
    const std::array<double, 4> distances = {
        std::hypot(x - ix0, y - iy0) * h, std::hypot(x - ix1, y - iy0) * h,
        std::hypot(x - ix0, y - iy1) * h, std::hypot(x - ix1, y - iy1) * h};
    return {ix0, iy0, ix1, iy1,
            {gid(ix0, iy0, ny), gid(ix1, iy0, ny), gid(ix0, iy1, ny), gid(ix1, iy1, ny)},
            wx, wy, weights, distances};
}

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
    const auto source = make_source_cell_2d(m, n, h, x, y);
    const int ix0 = source.ix0, iy0 = source.iy0, ix1 = source.ix1, iy1 = source.iy1;
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

static double bulk_adjoint_stencil(const double *T, const double *lam, const double *delta,
                                   int nx, int ny, int i, int j, double h) {
    auto T_at = [&](int ii, int jj) -> double {
        if (ii < 0 || ii >= nx || jj < 0 || jj >= ny) return 0.0;
        return T[gid(ii, jj, ny)];
    };
    auto L_at = [&](int ii, int jj) -> double {
        if (ii < 0 || ii >= nx || jj < 0 || jj >= ny) return 0.0;
        return lam[gid(ii, jj, ny)];
    };
    double a1m = 0, a1p = 0, a2m = 0, a2p = 0, b1m = 0, b1p = 0, b2m = 0, b2p = 0;
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
    const double coe = (a2p - a1m) / h + (b2p - b1m) / h;
    if (std::fabs(coe) < 1e-15) return 0.0;
    const double hadj = (a1p * L_at(i - 1, j) - a2m * L_at(i + 1, j)) / h +
                        (b1p * L_at(i, j - 1) - b2m * L_at(i, j + 1)) / h;
    return (delta[gid(i, j, ny)] + hadj) / coe;
}

static void solve_bulk_adjoint(double *lam, const double *T, const double *delta,
                               int nx, int ny, double h,
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
                        lam[gid(i, j, ny)] = bulk_adjoint_stencil(T, lam, delta, nx, ny, i, j, h);
            }
        double err = 0.0;
        for (int k = 0; k < nn; ++k) err = std::max(err, std::fabs(lam[k] - old[k]));
        if (err < tol) break;
    }
}

// Apply the exact transpose of the source-cell Simpson initialization.
static void apply_source_gradient(double *grad_f, const double *res,
                                  int m, int n, double h, double x, double y) {
    const auto source = make_source_cell_2d(m, n, h, x, y);
    for (int parameter_corner = 0; parameter_corner < 4; ++parameter_corner) {
        double value = 0.0;
        for (int travel_time_corner = 0; travel_time_corner < 4; ++travel_time_corner) {
            value += res[source.ids[travel_time_corner]] * source.distances[travel_time_corner] *
                     (source.weights[parameter_corner] + 1.0 +
                      (parameter_corner == travel_time_corner ? 1.0 : 0.0));
        }
        grad_f[source.ids[parameter_corner]] = value / 6.0;
    }
}

// Collect one Godunov Jacobian row as (col, val) pairs.
static void collect_godunov_row(std::vector<std::pair<int, double>> &entries, const double *u,
                                int m, int n, int i, int j, int ix0, int jx0, int ix1, int jx1) {
    entries.clear();
    const int ny = n + 1;
    int this_id = gid(i, j, ny);
    if (is_source_corner(i, j, ix0, jx0, ix1, jx1)) {
        entries.emplace_back(this_id, 1.0);
        return;
    }
    auto U = [&](int ii, int jj) { return u[gid(ii, jj, ny)]; };

    if (i == 0) {
        if (U(i, j) > U(i + 1, j)) {
            entries.emplace_back(this_id, 2 * (U(i, j) - U(i + 1, j)));
            entries.emplace_back(gid(i + 1, j, ny), 2 * (U(i + 1, j) - U(i, j)));
        }
    } else if (i == m) {
        if (U(i, j) > U(i - 1, j)) {
            entries.emplace_back(this_id, 2 * (U(i, j) - U(i - 1, j)));
            entries.emplace_back(gid(i - 1, j, ny), 2 * (U(i - 1, j) - U(i, j)));
        }
    } else {
        double a = U(i + 1, j) > U(i - 1, j) ? U(i - 1, j) : U(i + 1, j);
        if (U(i, j) > a) {
            entries.emplace_back(this_id, 2 * (U(i, j) - a));
            int nb = U(i + 1, j) > U(i - 1, j) ? gid(i - 1, j, ny) : gid(i + 1, j, ny);
            entries.emplace_back(nb, 2 * (a - U(i, j)));
        }
    }

    if (j == 0) {
        if (U(i, j) > U(i, j + 1)) {
            entries.emplace_back(this_id, 2 * (U(i, j) - U(i, j + 1)));
            entries.emplace_back(gid(i, j + 1, ny), 2 * (U(i, j + 1) - U(i, j)));
        }
    } else if (j == n) {
        if (U(i, j) > U(i, j - 1)) {
            entries.emplace_back(this_id, 2 * (U(i, j) - U(i, j - 1)));
            entries.emplace_back(gid(i, j - 1, ny), 2 * (U(i, j - 1) - U(i, j)));
        }
    } else {
        double b = U(i, j + 1) > U(i, j - 1) ? U(i, j - 1) : U(i, j + 1);
        if (U(i, j) > b) {
            entries.emplace_back(this_id, 2 * (U(i, j) - b));
            int nb = U(i, j + 1) > U(i, j - 1) ? gid(i, j - 1, ny) : gid(i, j + 1, ny);
            entries.emplace_back(nb, 2 * (b - U(i, j)));
        }
    }
}

// The pinned source-corner rows give an exact identity local system. Balance
// each corner against neighboring ordinary Godunov rows explicitly.
static void compute_source_corner_adjoint(double *res_out, const double *grad_u, const double *u,
                                          const double *lam_scaled, int m, int n, double x, double y,
                                          int radius = 1) {
    const int nx = m + 1, ny = n + 1, nn = nx * ny;
    const auto source = make_source_cell_2d(m, n, 1.0, x, y);
    const int ix0 = source.ix0, jx0 = source.iy0, ix1 = source.ix1, jx1 = source.iy1;

    int i_lo = std::max(0, ix0 - radius);
    int i_hi = std::min(m, ix1 + radius);
    int j_lo = std::max(0, jx0 - radius);
    int j_hi = std::min(n, jx1 + radius);

    std::memcpy(res_out, lam_scaled, sizeof(double) * nn);

    std::unordered_map<int, int> source_index;
    source_index.reserve(source.ids.size() * 2);
    for (int p = 0; p < 4; ++p) source_index[source.ids[p]] = p;
    for (int id : source.ids) res_out[id] = grad_u[id];

    std::vector<std::pair<int, double>> entries;
    entries.reserve(8);
    for (int i = i_lo; i <= i_hi; ++i) {
        for (int j = j_lo; j <= j_hi; ++j) {
            if (is_source_corner(i, j, ix0, jx0, ix1, jx1)) continue;
            const int row = gid(i, j, ny);
            collect_godunov_row(entries, u, m, n, i, j, ix0, jx0, ix1, jx1);
            for (const auto &entry : entries)
                if (source_index.find(entry.first) != source_index.end())
                    res_out[entry.first] -= entry.second * lam_scaled[row];
        }
    }
}

static void backward(double *grad_f, const double *grad_u, const double *u, const double *f,
                     int m, int n, double h, double x, double y) {
    const int nx = m + 1, ny = n + 1, nn = nx * ny;
    const double area = h * h;

    std::vector<double> delta(nn), lambda(nn), lam_scaled(nn), res(nn);
    for (int i = 0; i < nn; ++i) delta[i] = grad_u[i] / area;
    solve_bulk_adjoint(lambda.data(), u, delta.data(), nx, ny, h);
    for (int i = 0; i < nn; ++i) lam_scaled[i] = 0.5 * lambda[i];
    for (int i = 0; i < nn; ++i) grad_f[i] = lambda[i] * f[i] * area;

    compute_source_corner_adjoint(res.data(), grad_u, u, lam_scaled.data(), m, n, x, y, 1);
    apply_source_gradient(grad_f, res.data(), m, n, h, x, y);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &eikonal_forward, "Eikonal2D forward");
    m.def("backward", &eikonal_backward, "Eikonal2D backward");
}
