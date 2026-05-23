/* SPDX-License-Identifier: Apache-2.0
 *
 * Mirrors python/gdtw/utils.py.
 */
#pragma once

#include "types.hpp"

#include <cstddef>
#include <vector>

namespace gdtw {

/* Min-max rescale to [lo, hi]. Multi-channel input is rescaled per channel,
 * matching python/gdtw/utils.py::scale.
 */
std::vector<double> scale(const std::vector<double>& seq,
                          std::size_t d = 1,
                          double lo = -1.0,
                          double hi = 1.0);

/* linspace(a, b, n) — same as np.linspace; endpoints included. */
std::vector<double> linspace(double a, double b, std::size_t n);

/* np.interp: piecewise-linear, clamped at boundaries. xp must be increasing. */
double interp(double x, const std::vector<double>& xp, const std::vector<double>& fp);
std::vector<double> interp(const std::vector<double>& x,
                           const std::vector<double>& xp,
                           const std::vector<double>& fp);

/* Returns an elementwise penalty f: R -> R.
 *   make_penalty(Penalty::L1, _)    -> |x|
 *   make_penalty(Penalty::L2, _)    -> x*x
 *   make_penalty(Penalty::HUBER, d) -> 0.5 x^2 if |x| <= d else d*(|x| - 0.5 d)
 *   make_penalty(Penalty::CUSTOM, _)-> throws (caller must pass a PenaltyFn directly)
 */
PenaltyFn make_penalty(Penalty kind, double huber_delta = 1.0);

}  // namespace gdtw
