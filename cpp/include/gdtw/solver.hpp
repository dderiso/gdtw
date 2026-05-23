/* SPDX-License-Identifier: Apache-2.0
 *
 * DP kernel — bit-identical port of python/gdtw/gdtw.hpp::solve.
 * Pure function: takes precomputed t, Tau, D plus regularizers, returns the
 * optimal warping function tau[N], the discrete path[N], and the cost f_tau.
 */
#pragma once

#include "types.hpp"

#include <vector>

namespace gdtw {

struct SolveArgs {
    int N = 0;
    int M = 0;
    const std::vector<double>* t   = nullptr;    // length N
    const std::vector<double>* Tau = nullptr;    // N*M row-major
    const std::vector<double>* D   = nullptr;    // N*M row-major
    PenaltyFn R_cuml;
    PenaltyFn R_inst;
    double lambda_cuml = 1.0;
    double lambda_inst = 0.1;
    double s_min = 1e-8;
    double s_max = 1e8;
    bool BC_start_stop = true;
};

struct SolveResult {
    std::vector<double> tau;   // length N
    std::vector<int> path;     // length N
    double f_tau = 0.0;
};

SolveResult solve(const SolveArgs& args);

}  // namespace gdtw
