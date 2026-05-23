/* SPDX-License-Identifier: Apache-2.0
 *
 * Port of python/gdtw/solver.hpp::solve. The DP recurrence, the cumulative-cost
 * encoding (n[i,j] = D[i,j] + lambda_cuml * R_cuml(Tau[i,j] - t[i])), the
 * Bellman path-cost update, the boundary-condition handling, and the
 * backtrack are all kept identical so this implementation reproduces the
 * Python solver's outputs to numerical noise.
 */
#include "gdtw/solver.hpp"

#include <limits>
#include <stdexcept>

namespace gdtw {

namespace {
constexpr double DOUBLE_PRECISION_EPSILON = 1e-10;
constexpr double INF = std::numeric_limits<double>::infinity();
}  // namespace

SolveResult solve(const SolveArgs& a) {
    if (!a.t || !a.Tau || !a.D)
        throw GDTWError("solve: t, Tau, D must be non-null");
    if (!a.R_cuml || !a.R_inst)
        throw GDTWError("solve: R_cuml and R_inst must be set");
    if (a.M < 1 || (a.M % 2) == 0)
        throw GDTWError("solve: M must be a positive odd integer");

    const int N = a.N;
    const int M = a.M;
    const auto& t   = *a.t;
    const auto& Tau = *a.Tau;
    const auto& D   = *a.D;

    auto IDX = [M](int i, int j) { return i * M + j; };

    std::vector<double> f(static_cast<std::size_t>(N) * M, INF);
    std::vector<double> n(static_cast<std::size_t>(N) * M);
    std::vector<int>    p(static_cast<std::size_t>(N) * M, 0);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            n[IDX(i, j)] = D[IDX(i, j)] + a.lambda_cuml * a.R_cuml(Tau[IDX(i, j)] - t[i]);
        }
    }

    const int j_center = (M - 1) / 2 + 1;
    if (a.BC_start_stop) {
        for (int j = 0; j < M; ++j) f[IDX(0, j)] = INF;
        f[IDX(0, j_center)] = n[IDX(0, j_center)];
    } else {
        for (int j = 0; j < M; ++j) f[IDX(0, j)] = n[IDX(0, j)];
    }

    for (int i = 0; i < N - 1; ++i) {
        double dt = t[i + 1] - t[i];
        for (int j = 0; j < M; ++j) {
            for (int k = 0; k < M; ++k) {
                double slope = (Tau[IDX(i + 1, k)] - Tau[IDX(i, j)]) / dt;
                if (slope < (a.s_min - DOUBLE_PRECISION_EPSILON) ||
                    slope > (a.s_max + DOUBLE_PRECISION_EPSILON)) continue;
                double e_ijk = a.lambda_inst * a.R_inst(slope);
                double path_cost = f[IDX(i, j)] + dt * (e_ijk + n[IDX(i + 1, k)]);
                if (path_cost < f[IDX(i + 1, k)]) {
                    f[IDX(i + 1, k)] = path_cost;
                    p[IDX(i + 1, k)] = j;
                }
            }
        }
    }

    int j_opt = 0;
    if (a.BC_start_stop) {
        j_opt = j_center;
    } else {
        double mn = INF;
        for (int j = 0; j < M; ++j) {
            if (f[IDX(N - 1, j)] < mn) { mn = f[IDX(N - 1, j)]; j_opt = j; }
        }
    }

    SolveResult r;
    r.tau.assign(N, 0.0);
    r.path.assign(N, 0);
    r.f_tau = f[IDX(N - 1, j_opt)];
    for (int i = N - 1; i >= 0; --i) {
        r.tau[i]  = Tau[IDX(i, j_opt)];
        r.path[i] = j_opt;
        j_opt = p[IDX(i, j_opt)];
    }
    return r;
}

}  // namespace gdtw
