// 2D discrete adjoint with ordered (causal) solver

#include <torch/extension.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;

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

static void sweep_fsm_inf(double *u, const double *f, int nx, int ny, double h, int ix0, int iy0,
                          int ix1, int iy1, double tol = 1e-6, int max_iter = 200) {
    const int nn = nx * ny;
    auto T_at = [&](int i, int j) -> double {
        if (i < 0 || i >= nx || j < 0 || j >= ny) return FSM_INF;
        return u[gid(i, j, ny)];
    };
    auto is_source_corner = [&](int i, int j) {
        return (i == ix0 && j == iy0) || (i == ix1 && j == iy0) || (i == ix0 && j == iy1) ||
               (i == ix1 && j == iy1);
    };

    std::vector<double> old(nn);
    for (int it = 0; it < max_iter; ++it) {
        old.assign(u, u + nn);
        auto sweep = [&](int x0, int x1, int xd, int y0, int y1, int yd, auto left, auto down) {
            for (int j = y0; j != y1; j += yd) {
                for (int i = x0; i != x1; i += xd) {
                    if (is_source_corner(i, j)) continue;
                    const int k = gid(i, j, ny);
                    u[k] = std::min(u[k], godunov_update_2d(left(j, i), down(j, i), f[k], h, h));
                }
            }
        };
        sweep(0, nx, 1, 0, ny, 1, [&](int j, int i) { return i > 0 ? T_at(i - 1, j) : FSM_INF; },
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
        double d = std::hypot(x - ii, y - jj) * h;
        u[gid(ii, jj, ny)] = (d / 6.0) * (fsrc + 4.0 * fmid + fc);
    };
    set_corner(ix0, iy0, f00);
    set_corner(ix1, iy0, f10);
    set_corner(ix0, iy1, f01);
    set_corner(ix1, iy1, f11);
    sweep_fsm_inf(u, f, nx, ny, h, ix0, iy0, ix1, iy1);
}

// Assemble B (triplets) identical to Eikonal2D.cpp backward().
static void assemble_adjoint(std::vector<T> &triplets, Eigen::VectorXd &dFdf, const double *u,
                             const double *f, int m, int n, double h, int ix0, int jx0, int ix1,
                             int jx1) {
    const int ny = n + 1;
    const int N = (m + 1) * ny;
    dFdf.resize(N);
    for (int i = 0; i < N; ++i) dFdf[i] = -2.0 * f[i] * h * h;

    for (int i = 0; i < m + 1; ++i) {
        for (int j = 0; j < n + 1; ++j) {
            int idx = gid(i, j, ny);
            if ((i == ix0 && j == jx0) || (i == ix1 && j == jx0) || (i == ix0 && j == jx1) ||
                (i == ix1 && j == jx1)) {
                triplets.push_back(T(idx, idx, 1.0));
                continue;
            }

            if (i == 0) {
                if (u[idx] > u[gid(i + 1, j, ny)]) {
                    triplets.push_back(T(idx, idx, 2.0 * (u[idx] - u[gid(i + 1, j, ny)])));
                    triplets.push_back(T(idx, gid(i + 1, j, ny), 2.0 * (u[gid(i + 1, j, ny)] - u[idx])));
                }
            } else if (i == m) {
                if (u[idx] > u[gid(i - 1, j, ny)]) {
                    triplets.push_back(T(idx, idx, 2.0 * (u[idx] - u[gid(i - 1, j, ny)])));
                    triplets.push_back(T(idx, gid(i - 1, j, ny), 2.0 * (u[gid(i - 1, j, ny)] - u[idx])));
                }
            } else {
                double a = std::min(u[gid(i + 1, j, ny)], u[gid(i - 1, j, ny)]);
                if (u[idx] > a) {
                    triplets.push_back(T(idx, idx, 2.0 * (u[idx] - a)));
                    if (u[gid(i + 1, j, ny)] > u[gid(i - 1, j, ny)])
                        triplets.push_back(T(idx, gid(i - 1, j, ny), 2.0 * (a - u[idx])));
                    else
                        triplets.push_back(T(idx, gid(i + 1, j, ny), 2.0 * (a - u[idx])));
                }
            }

            if (j == 0) {
                if (u[idx] > u[gid(i, 1, ny)]) {
                    triplets.push_back(T(idx, idx, 2.0 * (u[idx] - u[gid(i, 1, ny)])));
                    triplets.push_back(T(idx, gid(i, 1, ny), 2.0 * (u[gid(i, 1, ny)] - u[idx])));
                }
            } else if (j == n) {
                if (u[idx] > u[gid(i, n - 1, ny)]) {
                    triplets.push_back(T(idx, idx, 2.0 * (u[idx] - u[gid(i, n - 1, ny)])));
                    triplets.push_back(T(idx, gid(i, n - 1, ny), 2.0 * (u[gid(i, n - 1, ny)] - u[idx])));
                }
            } else {
                double b = std::min(u[gid(i, j + 1, ny)], u[gid(i, j - 1, ny)]);
                if (u[idx] > b) {
                    triplets.push_back(T(idx, idx, 2.0 * (u[idx] - b)));
                    if (u[gid(i, j + 1, ny)] > u[gid(i, j - 1, ny)])
                        triplets.push_back(T(idx, gid(i, j - 1, ny), 2.0 * (b - u[idx])));
                    else
                        triplets.push_back(T(idx, gid(i, j + 1, ny), 2.0 * (b - u[idx])));
                }
            }
        }
    }
}

static void apply_simpson_source_grad(double *grad_f, const Eigen::VectorXd &res, double h, int m,
                                      int n, double x, double y, int ix0, int jx0, int ix1, int jx1) {
    const int ny = n + 1;
    double wx = x - ix0, wy = y - jx0;
    double w00 = (1 - wx) * (1 - wy), w10 = wx * (1 - wy);
    double w01 = (1 - wx) * wy, w11 = wx * wy;
    double res00 = res[gid(ix0, jx0, ny)], res10 = res[gid(ix1, jx0, ny)];
    double res01 = res[gid(ix0, jx1, ny)], res11 = res[gid(ix1, jx1, ny)];
    double d00 = std::hypot(x - ix0, y - jx0) * h;
    double d10 = std::hypot(x - ix1, y - jx0) * h;
    double d01 = std::hypot(x - ix0, y - jx1) * h;
    double d11 = std::hypot(x - ix1, y - jx1) * h;

    grad_f[gid(ix0, jx0, ny)] =
        (res00 * d00 * (w00 + 1 + 1) + res10 * d10 * (w00 + 1) + res01 * d01 * (w00 + 1) +
         res11 * d11 * (w00 + 1)) /
        6.0;
    grad_f[gid(ix1, jx0, ny)] =
        (res00 * d00 * (w10 + 1) + res10 * d10 * (w10 + 1 + 1) + res01 * d01 * (w10 + 1) +
         res11 * d11 * (w10 + 1)) /
        6.0;
    grad_f[gid(ix0, jx1, ny)] =
        (res00 * d00 * (w01 + 1) + res10 * d10 * (w01 + 1) + res01 * d01 * (w01 + 1 + 1) +
         res11 * d11 * (w01 + 1)) /
        6.0;
    grad_f[gid(ix1, jx1, ny)] =
        (res00 * d00 * (w11 + 1) + res10 * d10 * (w11 + 1) + res01 * d01 * (w11 + 1) +
         res11 * d11 * (w11 + 1 + 1)) /
        6.0;
}

static void backward_lu_impl(double *grad_f, const double *grad_u, const double *u, const double *f,
                             int m, int n, double h, double x, double y) {
    int ix0 = std::max(0, std::min((int)std::floor(x), m));
    int jx0 = std::max(0, std::min((int)std::floor(y), n));
    int ix1 = ix0 + 1, jx1 = jx0 + 1;
    const int N = (m + 1) * (n + 1);

    std::vector<T> triplets;
    Eigen::VectorXd dFdf;
    assemble_adjoint(triplets, dFdf, u, f, m, n, h, ix0, jx0, ix1, jx1);

    SpMat A(N, N);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A = A.transpose();
    Eigen::SparseLU<SpMat> solver;
    Eigen::Map<const Eigen::VectorXd> g(grad_u, N);
    solver.analyzePattern(A);
    solver.factorize(A);
    Eigen::VectorXd res = solver.solve(g);
    for (int i = 0; i < N; ++i) grad_f[i] = -res[i] * dFdf[i];
    apply_simpson_source_grad(grad_f, res, h, m, n, x, y, ix0, jx0, ix1, jx1);
}

// Causal back-substitution in descending traveltime order (same idea as 3D ordered).
static void backward_ordered_impl(double *grad_f, const double *grad_u, const double *u,
                                  const double *f, int m, int n, double h, double x, double y) {
    int ix0 = std::max(0, std::min((int)std::floor(x), m));
    int jx0 = std::max(0, std::min((int)std::floor(y), n));
    int ix1 = ix0 + 1, jx1 = jx0 + 1;
    const int N = (m + 1) * (n + 1);

    std::vector<T> triplets;
    Eigen::VectorXd dFdf;
    assemble_adjoint(triplets, dFdf, u, f, m, n, h, ix0, jx0, ix1, jx1);

    std::vector<std::vector<std::pair<int, double>>> col_entries(N);
    for (const auto &t : triplets) col_entries[t.col()].emplace_back(t.row(), t.value());

    std::vector<int> order(N);
    for (int i = 0; i < N; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return u[a] > u[b]; });

    Eigen::VectorXd res = Eigen::VectorXd::Zero(N);
    Eigen::Map<const Eigen::VectorXd> g(grad_u, N);
    for (int i : order) {
        double acc = g[i];
        double diag = 0.0;
        for (const auto &e : col_entries[i]) {
            if (e.first == i)
                diag += e.second;
            else
                acc -= e.second * res[e.first];
        }
        res[i] = diag != 0.0 ? acc / diag : 0.0;
    }

    for (int i = 0; i < N; ++i) grad_f[i] = -res[i] * dFdf[i];
    apply_simpson_source_grad(grad_f, res, h, m, n, x, y, ix0, jx0, ix1, jx1);
}

torch::Tensor eikonal_forward(torch::Tensor f, double h, double x, double y) {
    TORCH_CHECK(f.dim() == 2, "f should be 2D");
    TORCH_CHECK(f.is_contiguous(), "f should be contiguous");
    auto m = f.size(0) - 1, n = f.size(1) - 1;
    auto u = torch::zeros_like(f);
    forward(u.data_ptr<double>(), f.data_ptr<double>(), m, n, h, x, y);
    return u;
}

static torch::Tensor backward_dispatch(torch::Tensor grad_u, torch::Tensor u, torch::Tensor f,
                                       double h, double x, double y, bool ordered) {
    TORCH_CHECK(grad_u.dim() == 2, "grad_u should be 2D");
    TORCH_CHECK(grad_u.sizes() == u.sizes(), "grad_u and u should have the same size");
    TORCH_CHECK(grad_u.is_contiguous() && u.is_contiguous() && f.is_contiguous(),
                "tensors must be contiguous");
    auto m = f.size(0) - 1, n = f.size(1) - 1;
    auto grad_f = torch::zeros_like(f);
    if (ordered)
        backward_ordered_impl(grad_f.data_ptr<double>(), grad_u.data_ptr<double>(),
                              u.data_ptr<double>(), f.data_ptr<double>(), m, n, h, x, y);
    else
        backward_lu_impl(grad_f.data_ptr<double>(), grad_u.data_ptr<double>(), u.data_ptr<double>(),
                         f.data_ptr<double>(), m, n, h, x, y);
    return grad_f;
}

torch::Tensor eikonal_backward_lu(torch::Tensor grad_u, torch::Tensor u, torch::Tensor f, double h,
                                  double x, double y) {
    return backward_dispatch(grad_u, u, f, h, x, y, false);
}

torch::Tensor eikonal_backward_ordered(torch::Tensor grad_u, torch::Tensor u, torch::Tensor f,
                                       double h, double x, double y) {
    return backward_dispatch(grad_u, u, f, h, x, y, true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &eikonal_forward, "Eikonal2D forward (FSM)");
    m.def("backward", &eikonal_backward_ordered,
          "Eikonal2D backward (causal ordered, default)");
    m.def("backward_lu", &eikonal_backward_lu, "Eikonal2D backward (SparseLU)");
    m.def("backward_ordered", &eikonal_backward_ordered,
          "Eikonal2D backward (causal back-substitution)");
}
