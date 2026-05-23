/* SPDX-License-Identifier: Apache-2.0 */
#include "gdtw/warp.hpp"

namespace gdtw {

WarpResult warp(const GDTWParams& params) {
    auto g = std::make_shared<GDTW>();
    g->set_params(params).run();

    WarpResult r;
    r.g = g;
    r.f_tau = g->f_tau();
    /* x_f(tau) = x_signal_->evaluate_vector(tau[i]) for each i, flattened. */
    const auto& tau = g->tau();
    int N = g->N();
    std::size_t d = g->d();
    r.x_tau.assign(static_cast<std::size_t>(N) * d, 0.0);
    for (int i = 0; i < N; ++i) {
        auto v = g->signal_x().evaluate_vector(tau[i]);
        for (std::size_t k = 0; k < d; ++k) r.x_tau[i * d + k] = v[k];
    }
    /* phi(t) closure captures g by shared_ptr so the closure outlives `r.g` use cases. */
    auto g_capture = g;
    r.phi = [g_capture](double t) { return g_capture->phi(t); };
    return r;
}

WarpResult warp(const SignalSpec& x, const SignalSpec& y,
                const std::vector<double>& t, GDTWParams params) {
    params.x = x;
    params.y = y;
    if (!t.empty()) params.t = t;
    return warp(params);
}

}  // namespace gdtw
