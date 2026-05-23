/* SPDX-License-Identifier: Apache-2.0
 *
 * GDTW
 *
 * State machine wrapping the DP solver. Mirrors python/gdtw/gdtw.py.
 *
 * Process flow:
 *   1. set_params(p)       — stash hyperparameters and signal specs
 *   2. check_params()      — resolve N/M/d, build Signal x/y, build the
 *                            loss functional, build R_cum / R_inst
 *   3. allocate()          — allocate Tau (NxM), tau (N), path (N), D (NxM)
 *   4. iterate()           — loop max_iters times:
 *        compute_taus()       refine search corridor [l, u], fill Tau
 *        compute_dist_matrix() fill D = sum_k loss(x_k(Tau) - y_k(t))
 *        solve()              call solver::solve(...)
 *        early-stop if |f_tau - f_tau_prev| <= eps_abs + eps_rel * |f_tau_prev|
 *   5. run()               — convenience: check_params() -> allocate() -> iterate()
 *
 * Example:
 *   GDTWParams p; p.x = ...; p.y = ...;
 *   auto g = GDTW().set_params(p).run();
 *   double cost = g.f_tau();
 *   double mid = g.phi(0.5);
 */
#pragma once

#include "types.hpp"
#include "signal.hpp"
#include "solver.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

namespace gdtw {

enum class State { UNINITIALIZED, PARAMS_SET, CHECKED, ALLOCATED, DONE };

class GDTW {
public:
    GDTW() = default;

    GDTW& set_params(const GDTWParams& p);
    GDTW& check_params();
    GDTW& allocate();
    GDTW& compute_taus();
    GDTW& compute_dist_matrix();
    GDTW& solve();
    GDTW& iterate();
    GDTW& run();

    GDTWResult get_result() const;

    /* Accessors (only valid after iterate()/run()). */
    double phi(double t) const;
    double psi(double t) const;
    std::vector<double> get_psi_values() const;

    const std::vector<double>& t() const { return t_; }
    const std::vector<double>& tau() const { return tau_; }
    const std::vector<int>&    path() const { return path_; }
    const std::vector<double>& D() const { return D_; }
    double f_tau() const { return f_tau_; }
    int N() const { return N_; }
    int M() const { return M_; }
    std::size_t d() const { return d_; }
    int iteration() const { return iteration_; }
    State state() const { return state_; }
    const Signal& signal_x() const { return *x_signal_; }
    const Signal& signal_y() const { return *y_signal_; }

private:
    /* Helpers private by convention (per rules/CLASSES.md). */
    void _require_state(State expected, const char* method) const;
    void _build_loss();
    void _build_regularizers();

    GDTWParams params_{};
    State state_ = State::UNINITIALIZED;

    int N_ = -1;
    int M_ = -1;
    std::size_t d_ = 1;

    std::unique_ptr<Signal> x_signal_;
    std::unique_ptr<Signal> y_signal_;

    PenaltyFn loss_fn_;
    PenaltyFn r_cum_fn_;
    PenaltyFn r_inst_fn_;

    std::vector<double> t_;
    std::vector<double> u_;
    std::vector<double> l_;
    std::vector<double> u_orig_;
    std::vector<double> l_orig_;
    std::vector<double> Tau_;   // N*M row-major
    std::vector<double> D_;     // N*M row-major
    std::vector<double> tau_;   // N
    std::vector<int>    path_;  // N

    double f_tau_     = 0.0;
    double f_tau_prev_ = std::numeric_limits<double>::infinity();
    int iteration_   = 0;
};

}  // namespace gdtw
