// 3D discrete adjoint SparseLU — same structure as Eikonal2D.cpp.
// no source-corner (src) correction


#include <torch/extension.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <tuple>
#include <vector>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;

#define u(i, j, k) u[(i) * n * l + (j) * l + (k)]
#define f(i, j, k) f[(i) * n * l + (j) * l + (k)]
#define get_id(i, j, k) ((i) * n * l + (j) * l + (k))

static double calculate_unique_solution(double a1_, double a2_, double a3_, double fval, double h) {
    double a1 = a1_, a2 = a2_, a3 = a3_;
    if (a1 > a2) std::swap(a1, a2);
    if (a1 > a3) std::swap(a1, a3);
    if (a2 > a3) std::swap(a2, a3);

    double x = a1 + fval * h;
    if (x <= a2) return x;
    double B = -(a1 + a2);
    double C = (a1 * a1 + a2 * a2 - fval * fval * h * h) / 2.0;
    x = (-B + std::sqrt(B * B - 4 * C)) / 2.0;
    if (x <= a3) return x;
    B = -2.0 * (a1 + a2 + a3) / 3.0;
    C = (a1 * a1 + a2 * a2 + a3 * a3 - fval * fval * h * h) / 3.0;
    x = (-B + std::sqrt(B * B - 4 * C)) / 2.0;
    return x;
}

static void sweeping_over_I_J_K(double *u, const double *f, int m, int n, int l, double h,
                                int ix0, int jx0, int kx0, int ix1, int jx1, int kx1, int dirI,
                                int dirJ, int dirK) {
    auto I = std::make_tuple(dirI == 1 ? 0 : m - 1, dirI == 1 ? m : -1, dirI);
    auto J = std::make_tuple(dirJ == 1 ? 0 : n - 1, dirJ == 1 ? n : -1, dirJ);
    auto K = std::make_tuple(dirK == 1 ? 0 : l - 1, dirK == 1 ? l : -1, dirK);

    for (int i = std::get<0>(I); i != std::get<1>(I); i += std::get<2>(I))
        for (int j = std::get<0>(J); j != std::get<1>(J); j += std::get<2>(J))
            for (int k = std::get<0>(K); k != std::get<1>(K); k += std::get<2>(K)) {
                if (i == ix0 && j == jx0 && k == kx0) continue;
                if (i == ix0 && j == jx0 && k == kx1) continue;
                if (i == ix0 && j == jx1 && k == kx0) continue;
                if (i == ix0 && j == jx1 && k == kx1) continue;
                if (i == ix1 && j == jx0 && k == kx0) continue;
                if (i == ix1 && j == jx0 && k == kx1) continue;
                if (i == ix1 && j == jx1 && k == kx0) continue;
                if (i == ix1 && j == jx1 && k == kx1) continue;

                double uxmin = i == 0 ? u(i + 1, j, k)
                                      : (i == m - 1 ? u(i - 1, j, k)
                                                    : std::min(u(i + 1, j, k), u(i - 1, j, k)));
                double uymin = j == 0 ? u(i, j + 1, k)
                                      : (j == n - 1 ? u(i, j - 1, k)
                                                    : std::min(u(i, j + 1, k), u(i, j - 1, k)));
                double uzmin = k == 0 ? u(i, j, k + 1)
                                      : (k == l - 1 ? u(i, j, k - 1)
                                                    : std::min(u(i, j, k + 1), u(i, j, k - 1)));
                double u_new = calculate_unique_solution(uxmin, uymin, uzmin, f(i, j, k), h);
                u(i, j, k) = std::min(u_new, u(i, j, k));
            }
}

static void sweeping(double *u, const double *f, int m, int n, int l, double h, int ix0, int jx0,
                     int kx0, int ix1, int jx1, int kx1) {
    sweeping_over_I_J_K(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, 1, 1, 1);
    sweeping_over_I_J_K(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, -1, 1, 1);
    sweeping_over_I_J_K(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, -1, -1, 1);
    sweeping_over_I_J_K(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, 1, -1, 1);
    sweeping_over_I_J_K(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, 1, -1, -1);
    sweeping_over_I_J_K(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, 1, 1, -1);
    sweeping_over_I_J_K(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, -1, 1, -1);
    sweeping_over_I_J_K(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1, -1, -1, -1);
}

static void forward(double *u, const double *f, double h, int m, int n, int l, double x, double y,
                    double z, double tol = 1e-8) {
    int ix0 = std::max(0, std::min((int)std::floor(x), m - 1));
    int jx0 = std::max(0, std::min((int)std::floor(y), n - 1));
    int kx0 = std::max(0, std::min((int)std::floor(z), l - 1));
    int ix1 = ix0 + 1, jx1 = jx0 + 1, kx1 = kx0 + 1;
    for (int i = 0; i < m * n * l; i++) u[i] = 100000.0;

    double f000 = f(ix0, jx0, kx0), f001 = f(ix0, jx0, kx1);
    double f010 = f(ix0, jx1, kx0), f011 = f(ix0, jx1, kx1);
    double f100 = f(ix1, jx0, kx0), f101 = f(ix1, jx0, kx1);
    double f110 = f(ix1, jx1, kx0), f111 = f(ix1, jx1, kx1);

    double wx = x - ix0, wy = y - jx0, wz = z - kx0;
    double fsrc = (1 - wx) * (1 - wy) * (1 - wz) * f000 + (1 - wx) * (1 - wy) * wz * f001 +
                  (1 - wx) * wy * (1 - wz) * f010 + (1 - wx) * wy * wz * f011 +
                  wx * (1 - wy) * (1 - wz) * f100 + wx * (1 - wy) * wz * f101 +
                  wx * wy * (1 - wz) * f110 + wx * wy * wz * f111;
    double fmid = (f000 + f001 + f010 + f011 + f100 + f101 + f110 + f111) / 8.0;

    auto d = [&](int ii, int jj, int kk) {
        return std::sqrt((x - ii) * (x - ii) + (y - jj) * (y - jj) + (z - kk) * (z - kk)) * h;
    };
    u(ix0, jx0, kx0) = (d(ix0, jx0, kx0) / 6.0) * (fsrc + 4.0 * fmid + f000);
    u(ix0, jx0, kx1) = (d(ix0, jx0, kx1) / 6.0) * (fsrc + 4.0 * fmid + f001);
    u(ix0, jx1, kx0) = (d(ix0, jx1, kx0) / 6.0) * (fsrc + 4.0 * fmid + f010);
    u(ix0, jx1, kx1) = (d(ix0, jx1, kx1) / 6.0) * (fsrc + 4.0 * fmid + f011);
    u(ix1, jx0, kx0) = (d(ix1, jx0, kx0) / 6.0) * (fsrc + 4.0 * fmid + f100);
    u(ix1, jx0, kx1) = (d(ix1, jx0, kx1) / 6.0) * (fsrc + 4.0 * fmid + f101);
    u(ix1, jx1, kx0) = (d(ix1, jx1, kx0) / 6.0) * (fsrc + 4.0 * fmid + f110);
    u(ix1, jx1, kx1) = (d(ix1, jx1, kx1) / 6.0) * (fsrc + 4.0 * fmid + f111);

    auto u_old = new double[m * n * l];
    for (int i = 0; i < 20; i++) {
        std::memcpy(u_old, u, sizeof(double) * m * n * l);
        sweeping(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1);
        double err = 0.0;
        for (int j = 0; j < m * n * l; j++) err = std::max(std::fabs(u[j] - u_old[j]), err);
        if (err < tol) break;
    }
    delete[] u_old;
}

static void backward(double *grad_f, const double *grad_u, const double *u, const double *f,
                     double h, int m, int n, int l, double x, double y, double z) {
    // x,y,z kept for API parity with Eikonal3D.cpp (corners are not pinned / Simpson-free)
    (void)x;
    (void)y;
    (void)z;

    Eigen::VectorXd g(m * n * l);
    std::memcpy(g.data(), grad_u, sizeof(double) * m * n * l);

    Eigen::VectorXd rhs(m * n * l);
    for (int i = 0; i < m * n * l; i++) rhs[i] = -2 * f[i] * h * h;

    std::vector<T> triplets;
    std::set<int> zero_id;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < l; k++) {
                int this_id = get_id(i, j, k);

                // Diff vs LU: do NOT pin source corners with A_ii=1; assemble them too.

                double uxmin = i == 0 ? u(i + 1, j, k)
                                      : (i == m - 1 ? u(i - 1, j, k)
                                                    : std::min(u(i + 1, j, k), u(i - 1, j, k)));
                double uymin = j == 0 ? u(i, j + 1, k)
                                      : (j == n - 1 ? u(i, j - 1, k)
                                                    : std::min(u(i, j + 1, k), u(i, j - 1, k)));
                double uzmin = k == 0 ? u(i, j, k + 1)
                                      : (k == l - 1 ? u(i, j, k - 1)
                                                    : std::min(u(i, j, k + 1), u(i, j, k - 1)));

                int idx = i == 0 ? get_id(i + 1, j, k)
                                 : (i == m - 1 ? get_id(i - 1, j, k)
                                               : (u(i + 1, j, k) > u(i - 1, j, k) ? get_id(i - 1, j, k)
                                                                                  : get_id(i + 1, j, k)));
                int idy = j == 0 ? get_id(i, j + 1, k)
                                 : (j == n - 1 ? get_id(i, j - 1, k)
                                               : (u(i, j + 1, k) > u(i, j - 1, k) ? get_id(i, j - 1, k)
                                                                                  : get_id(i, j + 1, k)));
                int idz = k == 0 ? get_id(i, j, k + 1)
                                 : (k == l - 1 ? get_id(i, j, k - 1)
                                               : (u(i, j, k + 1) > u(i, j, k - 1) ? get_id(i, j, k - 1)
                                                                                  : get_id(i, j, k + 1)));

                bool this_id_is_not_zero = false;
                if (u(i, j, k) > uxmin) {
                    this_id_is_not_zero = true;
                    triplets.push_back(T(this_id, this_id, 2.0 * (u(i, j, k) - uxmin)));
                    triplets.push_back(T(this_id, idx, -2.0 * (u(i, j, k) - uxmin)));
                }
                if (u(i, j, k) > uymin) {
                    this_id_is_not_zero = true;
                    triplets.push_back(T(this_id, this_id, 2.0 * (u(i, j, k) - uymin)));
                    triplets.push_back(T(this_id, idy, -2.0 * (u(i, j, k) - uymin)));
                }
                if (u(i, j, k) > uzmin) {
                    this_id_is_not_zero = true;
                    triplets.push_back(T(this_id, this_id, 2.0 * (u(i, j, k) - uzmin)));
                    triplets.push_back(T(this_id, idz, -2.0 * (u(i, j, k) - uzmin)));
                }
                if (!this_id_is_not_zero) {
                    zero_id.insert(this_id);
                    g[this_id] = 0.0;
                }
            }
        }
    }

    if (zero_id.size() > 0) {
        printf("Warning: zero_id.size() = %zu\n", zero_id.size());
        for (auto &t : triplets) {
            if (zero_id.count(t.col()) || zero_id.count(t.row())) t = T(t.col(), t.row(), 0.0);
        }
        for (auto idx : zero_id) triplets.push_back(T(idx, idx, 1.0));
    }

    SpMat A(m * n * l, m * n * l);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A = A.transpose();
    Eigen::SparseLU<SpMat> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    Eigen::VectorXd res = solver.solve(g);
    for (int i = 0; i < m * n * l; i++) grad_f[i] = -res[i] * rhs[i];

    // Diff vs LU: no Simpson source-corner overwrite of grad_f.
}

torch::Tensor eikonal_forward(torch::Tensor f, double h, double x, double y, double z) {
    TORCH_CHECK(f.dim() == 3, "f must be a 3D tensor");
    TORCH_CHECK(f.is_contiguous(), "Input tensors must be contiguous");
    int m = f.size(0), n = f.size(1), l = f.size(2);
    auto u = torch::zeros_like(f);
    forward(u.data_ptr<double>(), f.data_ptr<double>(), h, m, n, l, x, y, z);
    return u;
}

torch::Tensor eikonal_backward(torch::Tensor grad_u, torch::Tensor u, torch::Tensor f, double h,
                               double x, double y, double z) {
    TORCH_CHECK(grad_u.dim() == 3 && u.dim() == 3 && f.dim() == 3, "All tensors must be 3D");
    TORCH_CHECK(grad_u.sizes() == u.sizes(), "All tensors must have the same size");
    TORCH_CHECK(grad_u.is_contiguous() && u.is_contiguous() && f.is_contiguous(),
                "All tensors must be contiguous");
    int m = u.size(0), n = u.size(1), l = u.size(2);
    auto grad_f = torch::zeros_like(f);
    backward(grad_f.data_ptr<double>(), grad_u.data_ptr<double>(), u.data_ptr<double>(),
             f.data_ptr<double>(), h, m, n, l, x, y, z);
    return grad_f;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &eikonal_forward, "3D forward");
    m.def("backward", &eikonal_backward, "3D global adjoint");
}
