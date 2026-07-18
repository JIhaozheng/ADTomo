// 3D discrete adjoint with ordered (causal) solver

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

#define U(i, j, k) u[(i) * n * l + (j) * l + (k)]
#define F(i, j, k) f[(i) * n * l + (j) * l + (k)]
#define GID(i, j, k) ((i) * n * l + (j) * l + (k))

// ---------------------------------------------------------------------------
// Forward FSM
// ---------------------------------------------------------------------------
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
                                int ix0, int jx0, int kx0, int ix1, int jx1, int kx1,
                                int dirI, int dirJ, int dirK) {
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
                U(i, j, k) = std::min(u_new, U(i, j, k));
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

    double f000 = F(ix0, jx0, kx0), f001 = F(ix0, jx0, kx1), f010 = F(ix0, jx1, kx0),
           f011 = F(ix0, jx1, kx1), f100 = F(ix1, jx0, kx0), f101 = F(ix1, jx0, kx1),
           f110 = F(ix1, jx1, kx0), f111 = F(ix1, jx1, kx1);

    double wx = x - ix0, wy = y - jx0, wz = z - kx0;
    double fsrc = (1 - wx) * (1 - wy) * (1 - wz) * f000 + (1 - wx) * (1 - wy) * wz * f001 +
                  (1 - wx) * wy * (1 - wz) * f010 + (1 - wx) * wy * wz * f011 +
                  wx * (1 - wy) * (1 - wz) * f100 + wx * (1 - wy) * wz * f101 +
                  wx * wy * (1 - wz) * f110 + wx * wy * wz * f111;
    double fmid = (f000 + f001 + f010 + f011 + f100 + f101 + f110 + f111) / 8.0;

    auto d = [&](int ii, int jj, int kk) {
        return std::sqrt((x - ii) * (x - ii) + (y - jj) * (y - jj) + (z - kk) * (z - kk)) * h;
    };
    U(ix0, jx0, kx0) = (d(ix0, jx0, kx0) / 6.0) * (fsrc + 4.0 * fmid + f000);
    U(ix0, jx0, kx1) = (d(ix0, jx0, kx1) / 6.0) * (fsrc + 4.0 * fmid + f001);
    U(ix0, jx1, kx0) = (d(ix0, jx1, kx0) / 6.0) * (fsrc + 4.0 * fmid + f010);
    U(ix0, jx1, kx1) = (d(ix0, jx1, kx1) / 6.0) * (fsrc + 4.0 * fmid + f011);
    U(ix1, jx0, kx0) = (d(ix1, jx0, kx0) / 6.0) * (fsrc + 4.0 * fmid + f100);
    U(ix1, jx0, kx1) = (d(ix1, jx0, kx1) / 6.0) * (fsrc + 4.0 * fmid + f101);
    U(ix1, jx1, kx0) = (d(ix1, jx1, kx0) / 6.0) * (fsrc + 4.0 * fmid + f110);
    U(ix1, jx1, kx1) = (d(ix1, jx1, kx1) / 6.0) * (fsrc + 4.0 * fmid + f111);

    auto u_old = new double[m * n * l];
    for (int it = 0; it < 20; it++) {
        std::memcpy(u_old, u, sizeof(double) * m * n * l);
        sweeping(u, f, m, n, l, h, ix0, jx0, kx0, ix1, jx1, kx1);
        double err = 0.0;
        for (int j = 0; j < m * n * l; j++) err = std::max(std::fabs(u[j] - u_old[j]), err);
        if (err < tol) break;
    }
    delete[] u_old;
}

// ---------------------------------------------------------------------------
// Shared adjoint assembly: build B (triplets), g (adjoint source), rhs (-2 f h^2).
// ---------------------------------------------------------------------------
static void assemble_adjoint(std::vector<T> &triplets, Eigen::VectorXd &g, Eigen::VectorXd &rhs,
                             const double *grad_u, const double *u, const double *f, double h,
                             int m, int n, int l, int ix0, int jx0, int kx0, int ix1, int jx1,
                             int kx1) {
    const int N = m * n * l;
    g.resize(N);
    std::memcpy(g.data(), grad_u, sizeof(double) * N);
    rhs.resize(N);
    for (int i = 0; i < N; i++) rhs[i] = -2.0 * f[i] * h * h;

    std::set<int> zero_id;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < l; k++) {
                int this_id = GID(i, j, k);
                if ((i == ix0 && j == jx0 && k == kx0) || (i == ix0 && j == jx0 && k == kx1) ||
                    (i == ix0 && j == jx1 && k == kx0) || (i == ix0 && j == jx1 && k == kx1) ||
                    (i == ix1 && j == jx0 && k == kx0) || (i == ix1 && j == jx0 && k == kx1) ||
                    (i == ix1 && j == jx1 && k == kx0) || (i == ix1 && j == jx1 && k == kx1)) {
                    triplets.push_back(T(this_id, this_id, 1.0));
                    continue;
                }

                double uxmin = i == 0 ? U(i + 1, j, k)
                                      : (i == m - 1 ? U(i - 1, j, k)
                                                    : std::min(U(i + 1, j, k), U(i - 1, j, k)));
                double uymin = j == 0 ? U(i, j + 1, k)
                                      : (j == n - 1 ? U(i, j - 1, k)
                                                    : std::min(U(i, j + 1, k), U(i, j - 1, k)));
                double uzmin = k == 0 ? U(i, j, k + 1)
                                      : (k == l - 1 ? U(i, j, k - 1)
                                                    : std::min(U(i, j, k + 1), U(i, j, k - 1)));

                int idx = i == 0 ? GID(i + 1, j, k)
                                 : (i == m - 1 ? GID(i - 1, j, k)
                                               : (U(i + 1, j, k) > U(i - 1, j, k) ? GID(i - 1, j, k)
                                                                                  : GID(i + 1, j, k)));
                int idy = j == 0 ? GID(i, j + 1, k)
                                 : (j == n - 1 ? GID(i, j - 1, k)
                                               : (U(i, j + 1, k) > U(i, j - 1, k) ? GID(i, j - 1, k)
                                                                                  : GID(i, j + 1, k)));
                int idz = k == 0 ? GID(i, j, k + 1)
                                 : (k == l - 1 ? GID(i, j, k - 1)
                                               : (U(i, j, k + 1) > U(i, j, k - 1) ? GID(i, j, k - 1)
                                                                                  : GID(i, j, k + 1)));

                bool active = false;
                if (U(i, j, k) > uxmin) {
                    active = true;
                    triplets.push_back(T(this_id, this_id, 2.0 * (U(i, j, k) - uxmin)));
                    triplets.push_back(T(this_id, idx, -2.0 * (U(i, j, k) - uxmin)));
                }
                if (U(i, j, k) > uymin) {
                    active = true;
                    triplets.push_back(T(this_id, this_id, 2.0 * (U(i, j, k) - uymin)));
                    triplets.push_back(T(this_id, idy, -2.0 * (U(i, j, k) - uymin)));
                }
                if (U(i, j, k) > uzmin) {
                    active = true;
                    triplets.push_back(T(this_id, this_id, 2.0 * (U(i, j, k) - uzmin)));
                    triplets.push_back(T(this_id, idz, -2.0 * (U(i, j, k) - uzmin)));
                }
                if (!active) {
                    zero_id.insert(this_id);
                    g[this_id] = 0.0;
                }
            }
        }
    }

    if (!zero_id.empty()) {
        printf("Warning: zero_id.size() = %zu\n", zero_id.size());
        for (auto &t : triplets) {
            if (zero_id.count(t.col()) || zero_id.count(t.row())) t = T(t.col(), t.row(), 0.0);
        }
        for (auto idx : zero_id) triplets.push_back(T(idx, idx, 1.0));
    }
}

// Simpson source-gradient term at the source box.
static void apply_simpson_source_grad(double *grad_f, const Eigen::VectorXd &res, double h, int m,
                                      int n, int l, double x, double y, double z, int ix0, int jx0,
                                      int kx0, int ix1, int jx1, int kx1) {
    double wx = x - ix0, wy = y - jx0, wz = z - kx0;
    double w000 = (1 - wx) * (1 - wy) * (1 - wz);
    double w001 = (1 - wx) * (1 - wy) * wz;
    double w010 = (1 - wx) * wy * (1 - wz);
    double w011 = (1 - wx) * wy * wz;
    double w100 = wx * (1 - wy) * (1 - wz);
    double w101 = wx * (1 - wy) * wz;
    double w110 = wx * wy * (1 - wz);
    double w111 = wx * wy * wz;

    double res000 = res[GID(ix0, jx0, kx0)], res001 = res[GID(ix0, jx0, kx1)];
    double res010 = res[GID(ix0, jx1, kx0)], res011 = res[GID(ix0, jx1, kx1)];
    double res100 = res[GID(ix1, jx0, kx0)], res101 = res[GID(ix1, jx0, kx1)];
    double res110 = res[GID(ix1, jx1, kx0)], res111 = res[GID(ix1, jx1, kx1)];

    auto dd = [&](int ii, int jj, int kk) {
        return std::sqrt((x - ii) * (x - ii) + (y - jj) * (y - jj) + (z - kk) * (z - kk)) * h;
    };
    double d000 = dd(ix0, jx0, kx0), d001 = dd(ix0, jx0, kx1), d010 = dd(ix0, jx1, kx0),
           d011 = dd(ix0, jx1, kx1), d100 = dd(ix1, jx0, kx0), d101 = dd(ix1, jx0, kx1),
           d110 = dd(ix1, jx1, kx0), d111 = dd(ix1, jx1, kx1);

    grad_f[GID(ix0, jx0, kx0)] =
        (res000 * d000 * (w000 + 1.5) + res001 * d001 * (w000 + 0.5) +
         res010 * d010 * (w000 + 0.5) + res011 * d011 * (w000 + 0.5) + res100 * d100 * (w000 + 0.5) +
         res101 * d101 * (w000 + 0.5) + res110 * d110 * (w000 + 0.5) + res111 * d111 * (w000 + 0.5)) /
        6.0;
    grad_f[GID(ix0, jx0, kx1)] =
        (res000 * d000 * (w001 + 0.5) + res001 * d001 * (w001 + 1.5) +
         res010 * d010 * (w001 + 0.5) + res011 * d011 * (w001 + 0.5) + res100 * d100 * (w001 + 0.5) +
         res101 * d101 * (w001 + 0.5) + res110 * d110 * (w001 + 0.5) + res111 * d111 * (w001 + 0.5)) /
        6.0;
    grad_f[GID(ix0, jx1, kx0)] =
        (res000 * d000 * (w010 + 0.5) + res001 * d001 * (w010 + 0.5) +
         res010 * d010 * (w010 + 1.5) + res011 * d011 * (w010 + 0.5) +
         res100 * d100 * (w010 + 0.5) + res101 * d101 * (w010 + 0.5) + res110 * d110 * (w010 + 0.5) +
         res111 * d111 * (w010 + 0.5)) /
        6.0;
    grad_f[GID(ix0, jx1, kx1)] =
        (res000 * d000 * (w011 + 0.5) + res001 * d001 * (w011 + 0.5) + res010 * d010 * (w011 + 0.5) +
         res011 * d011 * (w011 + 1.5) + res100 * d100 * (w011 + 0.5) +
         res101 * d101 * (w011 + 0.5) + res110 * d110 * (w011 + 0.5) + res111 * d111 * (w011 + 0.5)) /
        6.0;
    grad_f[GID(ix1, jx0, kx0)] =
        (res000 * d000 * (w100 + 0.5) + res001 * d001 * (w100 + 0.5) + res010 * d010 * (w100 + 0.5) +
         res011 * d011 * (w100 + 0.5) + res100 * d100 * (w100 + 1.5) +
         res101 * d101 * (w100 + 0.5) + res110 * d110 * (w100 + 0.5) + res111 * d111 * (w100 + 0.5)) /
        6.0;
    grad_f[GID(ix1, jx0, kx1)] =
        (res000 * d000 * (w101 + 0.5) + res001 * d001 * (w101 + 0.5) + res010 * d010 * (w101 + 0.5) +
         res011 * d011 * (w101 + 0.5) + res100 * d100 * (w101 + 0.5) +
         res101 * d101 * (w101 + 1.5) + res110 * d110 * (w101 + 0.5) +
         res111 * d111 * (w101 + 0.5)) /
        6.0;
    grad_f[GID(ix1, jx1, kx0)] =
        (res000 * d000 * (w110 + 0.5) + res001 * d001 * (w110 + 0.5) + res010 * d010 * (w110 + 0.5) +
         res011 * d011 * (w110 + 0.5) + res100 * d100 * (w110 + 0.5) + res101 * d101 * (w110 + 0.5) +
         res110 * d110 * (w110 + 1.5) + res111 * d111 * (w110 + 0.5)) /
        6.0;
    grad_f[GID(ix1, jx1, kx1)] =
        (res000 * d000 * (w111 + 0.5) + res001 * d001 * (w111 + 0.5) + res010 * d010 * (w111 + 0.5) +
         res011 * d011 * (w111 + 0.5) + res100 * d100 * (w111 + 0.5) + res101 * d101 * (w111 + 0.5) +
         res110 * d110 * (w111 + 0.5) + res111 * d111 * (w111 + 1.5)) /
        6.0;
}

// ---------------------------------------------------------------------------
// Solver 1: global SparseLU
// ---------------------------------------------------------------------------
static void backward_lu_impl(double *grad_f, const double *grad_u, const double *u, const double *f,
                             double h, int m, int n, int l, double x, double y, double z) {
    const int N = m * n * l;
    int ix0 = std::max(0, std::min((int)std::floor(x), m - 1));
    int jx0 = std::max(0, std::min((int)std::floor(y), n - 1));
    int kx0 = std::max(0, std::min((int)std::floor(z), l - 1));
    int ix1 = ix0 + 1, jx1 = jx0 + 1, kx1 = kx0 + 1;

    std::vector<T> triplets;
    Eigen::VectorXd g, rhs;
    assemble_adjoint(triplets, g, rhs, grad_u, u, f, h, m, n, l, ix0, jx0, kx0, ix1, jx1, kx1);

    SpMat A(N, N);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A = A.transpose();
    Eigen::SparseLU<SpMat> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    Eigen::VectorXd res = solver.solve(g);
    for (int i = 0; i < N; i++) grad_f[i] = -res[i] * rhs[i];

    apply_simpson_source_grad(grad_f, res, h, m, n, l, x, y, z, ix0, jx0, kx0, ix1, jx1, kx1);
}

// ---------------------------------------------------------------------------
// Solver 2: causal back-substitution in descending traveltime order.
// Upwind discretization => B is lower-triangular when nodes are ordered by ascending u, so
// B^T lambda = g is solved by a single sweep over nodes in descending u with no factorization.
//   lambda[i] = (g[i] - sum_{j != i} B[j,i] lambda[j]) / B[i,i]
// where col_entries[i] = {(j, B[j,i])}. Any (j, B[j,i]) with j != i has u[j] > u[i] (i is an
// upwind neighbor of j), so lambda[j] is already resolved earlier in the descending sweep.
// ---------------------------------------------------------------------------
static void backward_ordered_impl(double *grad_f, const double *grad_u, const double *u,
                                  const double *f, double h, int m, int n, int l, double x,
                                  double y, double z) {
    const int N = m * n * l;
    int ix0 = std::max(0, std::min((int)std::floor(x), m - 1));
    int jx0 = std::max(0, std::min((int)std::floor(y), n - 1));
    int kx0 = std::max(0, std::min((int)std::floor(z), l - 1));
    int ix1 = ix0 + 1, jx1 = jx0 + 1, kx1 = kx0 + 1;

    std::vector<T> triplets;
    Eigen::VectorXd g, rhs;
    assemble_adjoint(triplets, g, rhs, grad_u, u, f, h, m, n, l, ix0, jx0, kx0, ix1, jx1, kx1);

    std::vector<std::vector<std::pair<int, double>>> col_entries(N);
    for (const auto &t : triplets) col_entries[t.col()].emplace_back(t.row(), t.value());

    std::vector<int> order(N);
    for (int i = 0; i < N; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return u[a] > u[b]; });

    Eigen::VectorXd res = Eigen::VectorXd::Zero(N);
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

    for (int i = 0; i < N; i++) grad_f[i] = -res[i] * rhs[i];
    apply_simpson_source_grad(grad_f, res, h, m, n, l, x, y, z, ix0, jx0, kx0, ix1, jx1, kx1);
}

// ---------------------------------------------------------------------------
// PyTorch bindings
// ---------------------------------------------------------------------------
torch::Tensor eikonal_forward(torch::Tensor f, double h, double x, double y, double z) {
    TORCH_CHECK(f.dim() == 3, "f must be a 3D tensor");
    TORCH_CHECK(f.is_contiguous(), "Input tensors must be contiguous");
    int m = f.size(0), n = f.size(1), l = f.size(2);
    auto u = torch::zeros_like(f);
    forward(u.data_ptr<double>(), f.data_ptr<double>(), h, m, n, l, x, y, z);
    return u;
}

static torch::Tensor backward_dispatch(torch::Tensor grad_u, torch::Tensor u, torch::Tensor f,
                                       double h, double x, double y, double z, bool ordered) {
    TORCH_CHECK(grad_u.dim() == 3 && u.dim() == 3 && f.dim() == 3, "All tensors must be 3D");
    TORCH_CHECK(grad_u.sizes() == u.sizes(), "grad_u and u must match");
    TORCH_CHECK(grad_u.is_contiguous() && u.is_contiguous() && f.is_contiguous(),
                "All tensors must be contiguous");
    int m = u.size(0), n = u.size(1), l = u.size(2);
    auto grad_f = torch::zeros_like(f);
    if (ordered)
        backward_ordered_impl(grad_f.data_ptr<double>(), grad_u.data_ptr<double>(),
                              u.data_ptr<double>(), f.data_ptr<double>(), h, m, n, l, x, y, z);
    else
        backward_lu_impl(grad_f.data_ptr<double>(), grad_u.data_ptr<double>(), u.data_ptr<double>(),
                         f.data_ptr<double>(), h, m, n, l, x, y, z);
    return grad_f;
}

torch::Tensor eikonal_backward_lu(torch::Tensor grad_u, torch::Tensor u, torch::Tensor f, double h,
                                  double x, double y, double z) {
    return backward_dispatch(grad_u, u, f, h, x, y, z, false);
}

torch::Tensor eikonal_backward_ordered(torch::Tensor grad_u, torch::Tensor u, torch::Tensor f,
                                       double h, double x, double y, double z) {
    return backward_dispatch(grad_u, u, f, h, x, y, z, true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &eikonal_forward, "3D forward");
    m.def("backward", &eikonal_backward_lu, "3D SparseLU adjoint");
    m.def("backward_lu", &eikonal_backward_lu, "3D SparseLU adjoint");
    m.def("backward_ordered", &eikonal_backward_ordered, "3D ordered adjoint");
}
