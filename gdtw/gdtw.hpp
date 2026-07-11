/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2019-2026 Dave Deriso <dderiso@alumni.stanford.edu>, Twitter: @davederiso
 * Copyright (C) 2019-2024 Stephen Boyd
 *
 * GDTW is a library that performs dynamic time warping.
 * GDTW improves upon other methods (such as the original DTW, ShapeDTW, and FastDTW) by introducing regularization,
 * which obviates the need for pre-processing, and cross-validation for choosing optimal regularization hyper-parameters.
 *
 * Paper: https://rdcu.be/cT5dD
 * Source: https://github.com/dderiso/gdtw
 * Docs: https://dderiso.github.io/gdtw
 */

#pragma once

#include <vector>
#include <numeric>
#include <cmath>
#include <iterator>
#include <iostream>
#include <cfloat>
#include <limits>
#include <Python.h>
#include <functional>
#include <algorithm>

// A true IEEE infinity. Requires IEEE-conformant compares, so build with -O3
// (not -Ofast/-ffast-math, whose -ffinite-math-only makes comparisons against
// infinity undefined).
static const double GDTW_INF = std::numeric_limits<double>::infinity();

// loss functionals
#define L1_PENALTY(x) std::abs(x)
#define L2_PENALTY(x) (x)*(x)

// boundary conditions
#define DOUBLE_PRECISION_EPSILON 1e-10 // source of subtle errors -- really test this before changing
#define OUT_OF_BOUNDS(x, lower, upper) ((x < (lower - DOUBLE_PRECISION_EPSILON)) || (x > (upper + DOUBLE_PRECISION_EPSILON)))

/*
 * Penalty functors. The hot loop evaluates a regularizer O(N*M^2) times, so
 * the built-in penalties dispatch through an inlined switch on a small code
 * (no std::function indirection); user-supplied Python callables keep the
 * std::function path, which must hold the GIL (see gdtw_solver.cpp).
 */
enum GDTWPenaltyType { GDTW_PEN_L2 = 0, GDTW_PEN_L1 = 1, GDTW_PEN_HUBER = 2 };

struct GDTWPenaltyCode {
    int type;
    double delta;
    inline double operator()(const double& u) const {
        switch (type) {
            case GDTW_PEN_L2: return L2_PENALTY(u);
            case GDTW_PEN_L1: return L1_PENALTY(u);
            default: {
                const double au = std::abs(u);
                return au <= delta ? 0.5 * u * u : delta * (au - 0.5 * delta);
            }
        }
    }
};

struct GDTWPenaltyFn {
    const std::function<double(const double&)>* f;
    inline double operator()(const double& u) const { return (*f)(u); }
};

/*
 * The dynamic program.
 *
 * Objective convention (unchanged across releases). The recursion accumulates
 *     f = w_0 * n(0, j_0)  +  sum_{i=0}^{N-2} dt_i * ( e_ijk + n(i+1, k) ),
 * with node cost n(i,j) = D(i,j) + lambda_cuml * R_cuml(Tau(i,j) - t_i) and
 * edge cost e_ijk = lambda_inst * R_inst(slope) on the raw discrete slope: a
 * right-endpoint quadrature of the paper's discretized objective (its eq. (7)
 * is the left-endpoint rule; both are first-order consistent). With pinned
 * endpoints (BC_start_stop) the initial node is a path-constant and keeps
 * w_0 = 1, bit-compatible with every published release; with relaxed
 * boundaries the initial node's cost varies with the start choice, so it
 * carries the same quadrature weight as every other node (w_0 = dt_0): a
 * weight of 1 would overweight the start by a factor 1/dt_0 ~ N.
 *
 * Performance: rolling DP rows (f is two M-vectors; only the backpointer
 * table is O(N*M)); node costs computed once per stage; and because every
 * row of Tau is nondecreasing (a linspace between per-node bounds), the
 * band-feasible k's for fixed j form a contiguous window whose ends advance
 * monotonically with j -- found by two pointers with widened bounds, while
 * the original OUT_OF_BOUNDS test still decides each edge, so results are
 * identical to the full M*M scan. Unsorted rows (possible only for
 * infeasible bound configurations) fall back to the full scan.
 *
 * Returns 0 on success; 1 if no feasible path reached the terminal stage
 * under relaxed boundaries (previously an uninitialized read).
 */
template <class PenC, class PenI>
static int solve_impl(
    // inputs
    const int &N,
    const int &M,
    double* &t,
    double* &Tau,
    double* &D,

    // parameters
    PenC R_cuml,
    PenI R_inst,
    double &lambda_cuml,
    double &lambda_inst,
    double &s_min,
    double &s_max,
    bool &BC_start_stop,

    // outputs
    double* &tau,
    int* &path,
    double &f_of_tau
){
    double* f_prev = new double[M];
    double* f_next = new double[M];
    double* n_next = new double[M];
       int* P      = new    int[(size_t)N * M];

    // for readability
    #define   D_(i,j)   D[(size_t)(i)*M + (j)]
    #define Tau_(i,j) Tau[(size_t)(i)*M + (j)]
    #define   P_(i,j)   P[(size_t)(i)*M + (j)]

    int i,j,k;

    // init by filling i=0 (see the objective convention above)
    const int j_center = (M-1)/2; // M is always odd, so this is the exact 0-based center
    if (BC_start_stop){
        for (j=0; j<M; j++) f_prev[j] = GDTW_INF;
        f_prev[j_center] = D_(0,j_center) + lambda_cuml * R_cuml( Tau_(0,j_center) - t[0] ); // enforce t_0 = 0 (path-constant term, w_0 = 1)
    } else {
        const double dt0 = t[1] - t[0];
        for (j=0; j<M; j++) f_prev[j] = dt0 * (D_(0,j) + lambda_cuml * R_cuml( Tau_(0,j) - t[0] )); // relaxed start: w_0 = dt_0
    }

    for (i=0; i<N-1; i++){
        const double dt = t[i+1] - t[i];
        const double* row = &Tau_(i+1,0);
        bool row_sorted = true;
        for (k=0; k<M; k++){
            n_next[k] = D_(i+1,k) + lambda_cuml * R_cuml( row[k] - t[i+1] );
            f_next[k] = GDTW_INF;
            if (k > 0 && row[k] < row[k-1]) row_sorted = false;
        }
        int k_lo = 0, k_hi = 0;
        for (j=0; j<M; j++){
            const double tau_j = Tau_(i,j);
            if (row_sorted){
                // widened window bounds; advanced for every j to stay monotone
                const double lo = tau_j + (s_min - 2.0*DOUBLE_PRECISION_EPSILON) * dt;
                const double hi = tau_j + (s_max + 2.0*DOUBLE_PRECISION_EPSILON) * dt;
                while (k_lo < M && row[k_lo] <  lo) k_lo++;
                if (k_hi < k_lo) k_hi = k_lo;
                while (k_hi < M && row[k_hi] <= hi) k_hi++;
            } else {
                k_lo = 0; k_hi = M;
            }
            const double fj = f_prev[j];
            if (!(fj < GDTW_INF)) continue; // unreached state
            for (k=k_lo; k<k_hi; k++){
                const double slope = ( row[k] - tau_j ) / dt;
                if (OUT_OF_BOUNDS(slope, s_min, s_max)) continue; // boundary conditions
                const double e_ijk = lambda_inst * R_inst(slope); // edge cost
                const double path_cost = fj + dt * ( e_ijk + n_next[k] ); // Bellman
                if (path_cost < f_next[k]){
                    f_next[k] = path_cost; // min
                    P_(i+1,k) = j; // argmin
                }
            }
        }
        std::swap(f_prev, f_next);
    }

    // find terminal point of path
    int j_opt = -1;
    if(BC_start_stop){
        j_opt = j_center; // enforce t_N = 1
    }
    else {
        // argmin (unordered, linear search)
        double min = GDTW_INF;
        for (j=0; j<M; j++){
            if (f_prev[j] < min ){
                min  = f_prev[j];
                j_opt = j;
            }
        }
    }

    int status = 0;
    if (j_opt < 0){
        status = 1; // relaxed boundaries and no feasible path
    } else {
        // net cost
        f_of_tau = f_prev[j_opt];

        // re-trace path from terminal to origin point
        for (i=N-1; i>-1; i--){
            tau[i]  = Tau_(i,j_opt);
            path[i] = j_opt;
            if (i > 0) j_opt = P_(i,j_opt);
        }
    }

    delete[] f_prev;
    delete[] f_next;
    delete[] n_next;
    delete[] P;

    #undef D_
    #undef Tau_
    #undef P_
    return status;
}

// std::function form, kept for API stability with prior releases; the
// binding calls the code-dispatch instantiation directly when both
// regularizers are built-in strings.
inline int solve(
    const int &N,
    const int &M,
    double* &t,
    double* &Tau,
    double* &D,
    std::function<double(const double&)> &R_cuml,
    std::function<double(const double&)> &R_inst,
    double &lambda_cuml,
    double &lambda_inst,
    double &s_min,
    double &s_max,
    bool &BC_start_stop,
    double* &tau,
    int* &path,
    double &f_of_tau
){
    GDTWPenaltyFn pc{&R_cuml};
    GDTWPenaltyFn pi{&R_inst};
    return solve_impl(N, M, t, Tau, D, pc, pi, lambda_cuml, lambda_inst,
                      s_min, s_max, BC_start_stop, tau, path, f_of_tau);
}
