/* SPDX-License-Identifier: Apache-2.0 */
#include "gdtw/utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gdtw {

namespace {

std::vector<double> scale_1d(const double* p, std::size_t n, double lo, double hi) {
    double mn = std::numeric_limits<double>::infinity();
    double mx = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isnan(p[i])) continue;
        if (p[i] < mn) mn = p[i];
        if (p[i] > mx) mx = p[i];
    }
    double span = mx - mn;
    std::vector<double> out(n);
    if (span == 0.0) {
        std::fill(out.begin(), out.end(), lo);
        return out;
    }
    double a = hi - lo;
    for (std::size_t i = 0; i < n; ++i) out[i] = a * ((p[i] - mn) / span) + lo;
    return out;
}

}  // namespace

std::vector<double> scale(const std::vector<double>& seq, std::size_t d, double lo, double hi) {
    if (d <= 1) return scale_1d(seq.data(), seq.size(), lo, hi);
    std::size_t T = seq.size() / d;
    std::vector<double> col(T);
    std::vector<double> out(seq.size());
    for (std::size_t k = 0; k < d; ++k) {
        for (std::size_t i = 0; i < T; ++i) col[i] = seq[i * d + k];
        auto scaled = scale_1d(col.data(), T, lo, hi);
        for (std::size_t i = 0; i < T; ++i) out[i * d + k] = scaled[i];
    }
    return out;
}

std::vector<double> linspace(double a, double b, std::size_t n) {
    std::vector<double> out(n);
    if (n == 0) return out;
    if (n == 1) { out[0] = a; return out; }
    double step = (b - a) / static_cast<double>(n - 1);
    for (std::size_t i = 0; i < n; ++i) out[i] = a + step * static_cast<double>(i);
    return out;
}

double interp(double x, const std::vector<double>& xp, const std::vector<double>& fp) {
    if (xp.empty()) return std::numeric_limits<double>::quiet_NaN();
    if (x <= xp.front()) return fp.front();
    if (x >= xp.back())  return fp.back();
    auto it = std::upper_bound(xp.begin(), xp.end(), x);
    std::size_t j = static_cast<std::size_t>(it - xp.begin());
    double x0 = xp[j - 1], x1 = xp[j];
    double f0 = fp[j - 1], f1 = fp[j];
    double w = (x - x0) / (x1 - x0);
    return f0 + w * (f1 - f0);
}

std::vector<double> interp(const std::vector<double>& x,
                           const std::vector<double>& xp,
                           const std::vector<double>& fp) {
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = interp(x[i], xp, fp);
    return out;
}

PenaltyFn make_penalty(Penalty kind, double huber_delta) {
    switch (kind) {
        case Penalty::L1:
            return [](double x) { return std::abs(x); };
        case Penalty::L2:
            return [](double x) { return x * x; };
        case Penalty::HUBER: {
            const double d = huber_delta;
            return [d](double x) {
                double ax = std::abs(x);
                return ax <= d ? 0.5 * x * x : d * (ax - 0.5 * d);
            };
        }
        case Penalty::CUSTOM:
            throw GDTWError("make_penalty(CUSTOM): caller must supply a PenaltyFn directly");
    }
    throw GDTWError("make_penalty: unreachable");
}

}  // namespace gdtw
