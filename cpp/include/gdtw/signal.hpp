/* SPDX-License-Identifier: Apache-2.0
 *
 * Signal
 *
 * Mirrors python/gdtw/signal.py.
 * Accepts samples (1-D or 2-D row-major), an explicit (samples, t) pair, or a callable.
 * Optionally rescales every channel to [scale_lo, scale_hi], stores resampled
 * values on the working grid, and exposes a piecewise-linear evaluator.
 *
 * Process flow:
 *   1. ctor stashes the SignalSpec
 *   2. check_signal() resolves it to (z_a, z_t, d): array of samples + their
 *      timestamps + channel count; optionally rescales per-channel
 *   3. evaluate(t) / evaluate(ts) returns the interpolant at the given times
 *
 * Example:
 *   Signal s = Signal(z, "x", N).check_signal();
 *   double y = s.evaluate(0.5);            // scalar (d==1) or first channel
 *   auto v   = s.evaluate_vector(0.5);     // size-d vector
 *   auto Y   = s.evaluate_grid(t_query);   // (T_query * d) row-major
 */
#pragma once

#include "types.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace gdtw {

class Signal {
public:
    Signal(SignalSpec z, std::string name, int N,
           bool scale_signals = true, double scale_lo = -1.0, double scale_hi = 1.0,
           int verbose = 0);

    Signal& check_signal();

    std::size_t d() const { return d_; }
    const std::vector<double>& samples() const { return z_a_; }
    const std::vector<double>& sample_times() const { return t_z_; }

    /* Single-time evaluation: returns channel 0 (for d==1, the value). */
    double evaluate(double t) const;
    std::vector<double> evaluate_vector(double t) const;

    /* Grid evaluation: returns (ts.size() * d) row-major. For d==1 this is
     * just length ts.size().
     */
    std::vector<double> evaluate_grid(const std::vector<double>& ts) const;

private:
    SignalSpec z_;
    std::string name_;
    int N_;
    bool scale_signals_;
    double scale_lo_;
    double scale_hi_;
    int verbose_;  // reserved for future diagnostics; matches python/gdtw/signal.py

    std::vector<double> z_a_;  // (T * d) row-major
    std::vector<double> t_z_;  // length T
    std::size_t d_ = 1;
    bool checked_ = false;
};

}  // namespace gdtw
