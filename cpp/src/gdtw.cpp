/* SPDX-License-Identifier: Apache-2.0 */
#include "gdtw/gdtw.hpp"
#include "gdtw/utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <variant>

namespace gdtw {

namespace {
constexpr double INF = std::numeric_limits<double>::infinity();

bool spec_is_set(const SignalSpec& s) {
    return !std::holds_alternative<std::monostate>(s);
}
}  // namespace

GDTW& GDTW::set_params(const GDTWParams& p) {
    params_ = p;
    state_ = State::PARAMS_SET;
    return *this;
}

void GDTW::_require_state(State expected, const char* method) const {
    if (static_cast<int>(state_) < static_cast<int>(expected)) {
        throw GDTWError(std::string("GDTW::") + method + ": invalid state (call earlier methods first)");
    }
}

void GDTW::_build_loss() {
    if (params_.loss == Penalty::CUSTOM) {
        if (!params_.custom_loss)
            throw GDTWError("Loss=CUSTOM requires GDTWParams::custom_loss to be set");
        loss_fn_ = params_.custom_loss;
    } else {
        loss_fn_ = make_penalty(params_.loss, params_.huber_delta);
    }
}

void GDTW::_build_regularizers() {
    r_cum_fn_  = make_penalty(params_.r_cum,  params_.huber_delta);
    r_inst_fn_ = make_penalty(params_.r_inst, params_.huber_delta);
}

GDTW& GDTW::check_params() {
    _require_state(State::PARAMS_SET, "check_params");

    /* Decide N. Three branches mirror gdtw.py::check_params. */
    if (!params_.t.empty()) {
        t_ = params_.t;
        if (params_.N <= 0) {
            N_ = static_cast<int>(t_.size());
        } else if (params_.N == static_cast<int>(t_.size())) {
            N_ = params_.N;
        } else {
            /* Disagreement. Python picks the smaller, except when t is
             * irregularly sampled in which case it keeps t. We approximate
             * by always preferring t when sizes mismatch — simpler and the
             * Python behavior on regularly-spaced t collapses to the same.
             */
            N_ = static_cast<int>(t_.size());
        }
    } else if (params_.N > 0) {
        N_ = params_.N;
        t_ = linspace(0.0, 1.0, static_cast<std::size_t>(N_));
    } else {
        /* Neither t nor N set: take length from x or y if they're arrays. */
        if (auto* s = std::get_if<SignalSamples>(&params_.x)) {
            N_ = static_cast<int>(s->data.size() / std::max<std::size_t>(s->d, 1));
        } else if (auto* s = std::get_if<SignalSamples>(&params_.y)) {
            N_ = static_cast<int>(s->data.size() / std::max<std::size_t>(s->d, 1));
        } else if (auto* s = std::get_if<SignalSamplesWithT>(&params_.x)) {
            N_ = static_cast<int>(s->samples.data.size() / std::max<std::size_t>(s->samples.d, 1));
        } else if (auto* s = std::get_if<SignalSamplesWithT>(&params_.y)) {
            N_ = static_cast<int>(s->samples.data.size() / std::max<std::size_t>(s->samples.d, 1));
        } else {
            N_ = params_.N_default;
        }
        t_ = linspace(0.0, 1.0, static_cast<std::size_t>(N_));
    }

    /* M auto-sizing: M = min(int(0.55*N), M_max), then forced odd. */
    if (params_.M <= 0 || params_.M >= N_) {
        double cand = std::min(N_ * 0.55, static_cast<double>(params_.M_max));
        M_ = static_cast<int>(cand);
    } else {
        M_ = params_.M;
    }
    if (M_ % 2 == 0) ++M_;

    /* Signals. */
    if (!spec_is_set(params_.x)) throw GDTWError("Signal x is missing.");
    if (!spec_is_set(params_.y)) throw GDTWError("Signal y is missing.");
    x_signal_ = std::make_unique<Signal>(params_.x, "x", N_,
        params_.scale_signals, -1.0, 1.0, params_.verbose);
    y_signal_ = std::make_unique<Signal>(params_.y, "y", N_,
        params_.scale_signals, -1.0, 1.0, params_.verbose);
    x_signal_->check_signal();
    y_signal_->check_signal();
    if (x_signal_->d() != y_signal_->d()) {
        throw GDTWError("Signal channel mismatch: x has d=" +
                        std::to_string(x_signal_->d()) +
                        " channels but y has d=" + std::to_string(y_signal_->d()) +
                        ". A single warping function aligns both signals, so x and y must have the same number of channels.");
    }
    d_ = x_signal_->d();

    _build_loss();
    _build_regularizers();

    state_ = State::CHECKED;
    return *this;
}

GDTW& GDTW::allocate() {
    _require_state(State::CHECKED, "allocate");
    Tau_.assign(static_cast<std::size_t>(N_) * M_, 0.0);
    D_.assign(static_cast<std::size_t>(N_) * M_, 0.0);
    tau_.assign(N_, 0.0);
    path_.assign(N_, 0);
    u_.clear(); l_.clear(); u_orig_.clear(); l_orig_.clear();
    f_tau_ = 0.0;
    f_tau_prev_ = INF;
    iteration_ = 0;
    state_ = State::ALLOCATED;
    return *this;
}

GDTW& GDTW::compute_taus() {
    _require_state(State::ALLOCATED, "compute_taus");
    if (iteration_ == 0) {
        u_.resize(N_); l_.resize(N_);
        for (int i = 0; i < N_; ++i) {
            double a1 = params_.s_beta + params_.s_max * t_[i];
            double a2 = params_.s_beta + 1.0 - params_.s_min * (1.0 - t_[i]);
            double b1 = params_.s_min * t_[i];
            double b2 = -params_.s_beta + 1.0 - params_.s_max * (1.0 - t_[i]);
            u_[i] = std::min({a1, a2, 1.0});
            l_[i] = std::max({b1, b2, 0.0});
        }
        u_orig_ = u_;
        l_orig_ = l_;
    } else {
        for (int i = 0; i < N_; ++i) {
            double tau_range = params_.eta * (u_[i] - l_[i]) / 2.0;
            u_[i] = std::min(tau_[i] + tau_range, u_orig_[i]);
            l_[i] = std::max(tau_[i] - tau_range, l_orig_[i]);
        }
    }
    for (int i = 0; i < N_; ++i) {
        double base = l_[i];
        double span = u_[i] - l_[i];
        double denom = static_cast<double>(M_ - 1);
        for (int j = 0; j < M_; ++j) {
            Tau_[i * M_ + j] = base + span * (static_cast<double>(j) / denom);
        }
    }
    return *this;
}

GDTW& GDTW::compute_dist_matrix() {
    _require_state(State::ALLOCATED, "compute_dist_matrix");

    /* D[i,j] = sum_k loss(x_k(Tau[i,j]) - y_k(target_k(i,j))).
     * In asymmetric mode target = t[i] (independent of j).
     * In symmetric mode target = 2*t[i] - Tau[i,j].
     */
    std::vector<double> y_at_t;
    if (!params_.symmetric) {
        y_at_t = y_signal_->evaluate_grid(t_);  // (N * d)
    }

    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < M_; ++j) {
            double tau_ij = Tau_[i * M_ + j];
            double sum = 0.0;
            bool out_of_range = false;
            for (std::size_t k = 0; k < d_; ++k) {
                /* x_k at tau_ij */
                double xk;
                {
                    auto v = x_signal_->evaluate_vector(tau_ij);
                    xk = v[k];
                }
                /* y_k at target */
                double yk;
                if (params_.symmetric) {
                    double psi_ij = 2.0 * t_[i] - tau_ij;
                    if (psi_ij < 0.0 || psi_ij > 1.0) out_of_range = true;
                    auto v = y_signal_->evaluate_vector(psi_ij);
                    yk = v[k];
                } else {
                    yk = y_at_t[i * d_ + k];
                }
                if (std::isnan(xk)) xk = INF;
                if (std::isnan(yk)) yk = INF;
                sum += loss_fn_(xk - yk);
            }
            D_[i * M_ + j] = out_of_range ? INF : sum;
        }
    }
    return *this;
}

GDTW& GDTW::solve() {
    _require_state(State::ALLOCATED, "solve");
    SolveArgs a;
    a.N = N_; a.M = M_;
    a.t = &t_; a.Tau = &Tau_; a.D = &D_;
    a.R_cuml = r_cum_fn_;
    a.R_inst = r_inst_fn_;
    a.lambda_cuml = params_.lambda_cum;
    a.lambda_inst = params_.lambda_inst;
    a.s_min = params_.s_min;
    a.s_max = params_.s_max;
    a.BC_start_stop = params_.BC_start_stop;
    SolveResult r = ::gdtw::solve(a);
    tau_  = std::move(r.tau);
    path_ = std::move(r.path);
    f_tau_ = r.f_tau;
    return *this;
}

GDTW& GDTW::iterate() {
    _require_state(State::ALLOCATED, "iterate");
    for (iteration_ = 0; iteration_ < params_.max_iters; ++iteration_) {
        compute_taus();
        compute_dist_matrix();
        solve();
        if (iteration_ > 0 && f_tau_prev_ != INF) {
            if (std::abs(f_tau_ - f_tau_prev_) <=
                params_.epsilon_abs + params_.epsilon_rel * std::abs(f_tau_prev_)) {
                break;
            }
        }
        f_tau_prev_ = f_tau_;
    }
    state_ = State::DONE;
    return *this;
}

GDTW& GDTW::run() {
    check_params();
    allocate();
    iterate();
    return *this;
}

double GDTW::phi(double t) const {
    _require_state(State::DONE, "phi");
    return interp(t, t_, tau_);
}

double GDTW::psi(double t) const {
    return 2.0 * t - phi(t);
}

std::vector<double> GDTW::get_psi_values() const {
    _require_state(State::DONE, "get_psi_values");
    std::vector<double> out(static_cast<std::size_t>(N_));
    for (int i = 0; i < N_; ++i) out[i] = 2.0 * t_[i] - tau_[i];
    return out;
}

GDTWResult GDTW::get_result() const {
    _require_state(State::DONE, "get_result");
    GDTWResult r;
    r.t = t_;
    r.tau = tau_;
    r.x = x_signal_->samples();
    r.y = y_signal_->samples();
    r.x_hat.assign(static_cast<std::size_t>(N_) * d_, 0.0);
    for (int i = 0; i < N_; ++i) {
        auto v = x_signal_->evaluate_vector(tau_[i]);
        for (std::size_t k = 0; k < d_; ++k) r.x_hat[i * d_ + k] = v[k];
    }
    r.path = path_;
    r.f_tau = f_tau_;
    r.iterations = iteration_;
    r.N = N_;
    r.M = M_;
    r.d = static_cast<int>(d_);
    return r;
}

}  // namespace gdtw
