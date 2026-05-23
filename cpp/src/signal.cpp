/* SPDX-License-Identifier: Apache-2.0 */
#include "gdtw/signal.hpp"
#include "gdtw/utils.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <variant>

namespace gdtw {

namespace {

bool needs_scaling(const std::vector<double>& z, std::size_t d,
                   double lo, double hi) {
    std::size_t T = z.size() / d;
    for (std::size_t k = 0; k < d; ++k) {
        double mn = std::numeric_limits<double>::infinity();
        double mx = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < T; ++i) {
            double v = z[i * d + k];
            if (std::isnan(v)) continue;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        if (mn != lo || mx != hi) return true;
    }
    return false;
}

double interp_channel(double t, const std::vector<double>& t_z,
                      const std::vector<double>& z_a, std::size_t d, std::size_t k) {
    if (t_z.empty()) return std::numeric_limits<double>::quiet_NaN();
    if (t <= t_z.front()) return z_a[k];
    if (t >= t_z.back())  return z_a[(t_z.size() - 1) * d + k];
    auto it = std::upper_bound(t_z.begin(), t_z.end(), t);
    std::size_t j = static_cast<std::size_t>(it - t_z.begin());
    double t0 = t_z[j - 1], t1 = t_z[j];
    double f0 = z_a[(j - 1) * d + k], f1 = z_a[j * d + k];
    double w = (t - t0) / (t1 - t0);
    return f0 + w * (f1 - f0);
}

}  // namespace

Signal::Signal(SignalSpec z, std::string name, int N,
               bool scale_signals, double scale_lo, double scale_hi, int verbose)
    : z_(std::move(z)),
      name_(std::move(name)),
      N_(N),
      scale_signals_(scale_signals),
      scale_lo_(scale_lo),
      scale_hi_(scale_hi),
      verbose_(verbose) {}

Signal& Signal::check_signal() {
    /* Resolve every SignalSpec form into (z_a_, t_z_, d_).
     * Callable specs are sampled at linspace(0,1,N) per Python's behavior.
     */
    if (std::holds_alternative<std::monostate>(z_)) {
        throw GDTWError("Signal " + name_ + " is missing. Both signals (x and y) are required.");
    }

    auto sample_callable_scalar = [&](const SignalCallableScalar& f) {
        t_z_ = linspace(0.0, 1.0, static_cast<std::size_t>(N_));
        z_a_.resize(t_z_.size());
        d_ = 1;
        for (std::size_t i = 0; i < t_z_.size(); ++i) z_a_[i] = f(t_z_[i]);
    };

    auto sample_callable_vector = [&](const SignalCallableVector& f) {
        t_z_ = linspace(0.0, 1.0, static_cast<std::size_t>(N_));
        std::vector<double> probe = f(t_z_.front());
        d_ = probe.size();
        if (d_ == 0) throw GDTWError("Signal " + name_ + ": callable returned zero-length vector");
        z_a_.assign(t_z_.size() * d_, 0.0);
        for (std::size_t k = 0; k < d_; ++k) z_a_[k] = probe[k];
        for (std::size_t i = 1; i < t_z_.size(); ++i) {
            std::vector<double> v = f(t_z_[i]);
            if (v.size() != d_)
                throw GDTWError("Signal " + name_ + ": callable returned inconsistent dim");
            for (std::size_t k = 0; k < d_; ++k) z_a_[i * d_ + k] = v[k];
        }
    };

    if (auto* f = std::get_if<SignalCallableScalar>(&z_)) {
        sample_callable_scalar(*f);
    } else if (auto* f = std::get_if<SignalCallableVector>(&z_)) {
        sample_callable_vector(*f);
    } else if (auto* s = std::get_if<SignalSamples>(&z_)) {
        if (s->d == 0) throw GDTWError("Signal " + name_ + ": d must be >= 1");
        d_ = s->d;
        z_a_ = s->data;
        std::size_t T = z_a_.size() / d_;
        if (T * d_ != z_a_.size())
            throw GDTWError("Signal " + name_ + ": length not divisible by d");
        t_z_ = linspace(0.0, 1.0, T);
    } else if (auto* s = std::get_if<SignalSamplesWithT>(&z_)) {
        if (s->samples.d == 0) throw GDTWError("Signal " + name_ + ": d must be >= 1");
        d_ = s->samples.d;
        z_a_ = s->samples.data;
        std::size_t T = z_a_.size() / d_;
        if (T * d_ != z_a_.size())
            throw GDTWError("Signal " + name_ + ": length not divisible by d");
        if (s->t.size() != T)
            throw GDTWError("Signal " + name_ + ": (samples, t) length mismatch");
        t_z_ = s->t;
    }

    if (scale_signals_ && !z_a_.empty()) {
        if (needs_scaling(z_a_, d_, scale_lo_, scale_hi_)) {
            z_a_ = scale(z_a_, d_, scale_lo_, scale_hi_);
        }
    }

    checked_ = true;
    return *this;
}

double Signal::evaluate(double t) const {
    if (!checked_) throw GDTWError("Signal::evaluate before check_signal()");
    return interp_channel(t, t_z_, z_a_, d_, 0);
}

std::vector<double> Signal::evaluate_vector(double t) const {
    if (!checked_) throw GDTWError("Signal::evaluate_vector before check_signal()");
    std::vector<double> out(d_);
    for (std::size_t k = 0; k < d_; ++k) out[k] = interp_channel(t, t_z_, z_a_, d_, k);
    return out;
}

std::vector<double> Signal::evaluate_grid(const std::vector<double>& ts) const {
    if (!checked_) throw GDTWError("Signal::evaluate_grid before check_signal()");
    std::vector<double> out(ts.size() * d_);
    for (std::size_t i = 0; i < ts.size(); ++i) {
        for (std::size_t k = 0; k < d_; ++k) {
            out[i * d_ + k] = interp_channel(ts[i], t_z_, z_a_, d_, k);
        }
    }
    return out;
}

}  // namespace gdtw
