/* SPDX-License-Identifier: Apache-2.0
 *
 * High-level entry point. Mirrors python/gdtw/warp.py::warp.
 */
#pragma once

#include "gdtw.hpp"
#include "types.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace gdtw {

struct WarpResult {
    std::function<double(double)> phi;       // phi(t)
    std::vector<double> x_tau;               // x_f(tau) flattened (N*d)
    double f_tau = 0.0;
    std::shared_ptr<GDTW> g;
};

/* Convenience overloads. Equivalent to:
 *     GDTW g; g.set_params(params).run();
 *     return { phi=lambda t: g.phi(t), x_tau=g.x_f(g.tau), f_tau=g.f_tau, g }
 */
WarpResult warp(const GDTWParams& params);
WarpResult warp(const SignalSpec& x, const SignalSpec& y,
                const std::vector<double>& t,
                GDTWParams params = {});

}  // namespace gdtw
