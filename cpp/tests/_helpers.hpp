/* SPDX-License-Identifier: Apache-2.0
 *
 * Helpers for the C++ test suite. Mirrors python/test/_helpers.py.
 */
#pragma once

#include "gdtw/types.hpp"
#include "gdtw/utils.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <vector>

namespace test_helpers {

inline double phi_true(double t) { return t * t; }

struct Pair1D {
    std::vector<double> t;
    std::vector<double> x;
    std::vector<double> y;
};

struct PairMD {
    std::vector<double> t;
    std::vector<double> x;  // (T * d)
    std::vector<double> y;  // (T * d)
    std::size_t d = 1;
};

inline Pair1D scalar_pair(int T = 200, double freq = 5.0) {
    Pair1D p;
    p.t = gdtw::linspace(0.0, 1.0, static_cast<std::size_t>(T));
    p.x.resize(T); p.y.resize(T);
    for (int i = 0; i < T; ++i) {
        p.x[i] = std::sin(2.0 * M_PI * freq * p.t[i]);
        p.y[i] = std::sin(2.0 * M_PI * freq * phi_true(p.t[i]));
    }
    return p;
}

/* multid_pair with explicit frequencies. If freqs is empty, generates d
 * independent frequencies from a seeded PRNG matching the spirit of
 * numpy.random.default_rng(seed).uniform(0, 6).
 */
inline PairMD multid_pair(std::size_t d, int T = 300,
                          std::vector<double> freqs = {}, unsigned seed = 0) {
    if (freqs.empty()) {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> u(0.0, 6.0);
        freqs.resize(d);
        for (std::size_t k = 0; k < d; ++k) freqs[k] = 2.0 + u(rng);
    }
    PairMD p;
    p.d = d;
    p.t = gdtw::linspace(0.0, 1.0, static_cast<std::size_t>(T));
    p.x.assign(static_cast<std::size_t>(T) * d, 0.0);
    p.y.assign(static_cast<std::size_t>(T) * d, 0.0);
    for (int i = 0; i < T; ++i) {
        for (std::size_t k = 0; k < d; ++k) {
            p.x[i * d + k] = std::sin(2.0 * M_PI * freqs[k] * p.t[i]);
            p.y[i * d + k] = std::sin(2.0 * M_PI * freqs[k] * phi_true(p.t[i]));
        }
    }
    return p;
}

/* Dispatch: d==1 -> scalar_pair, d>=2 -> multid_pair. Returns PairMD. */
inline PairMD make_pair(std::size_t d, int T = 300, std::vector<double> freqs = {}) {
    if (d == 1) {
        Pair1D s = scalar_pair(T);
        PairMD p;
        p.d = 1;
        p.t = std::move(s.t);
        p.x = std::move(s.x);
        p.y = std::move(s.y);
        return p;
    }
    return multid_pair(d, T, std::move(freqs));
}

inline double linf_phi_error(const std::function<double(double)>& phi,
                             const std::vector<double>& t) {
    double mx = 0.0;
    for (double ti : t) {
        double e = std::abs(phi(ti) - phi_true(ti));
        if (e > mx) mx = e;
    }
    return mx;
}

inline gdtw::SignalSpec spec_from(const std::vector<double>& data, std::size_t d) {
    gdtw::SignalSamples s; s.data = data; s.d = d;
    return s;
}

inline gdtw::SignalSpec spec_from_with_t(const std::vector<double>& data, std::size_t d,
                                         const std::vector<double>& t) {
    gdtw::SignalSamplesWithT s; s.samples.data = data; s.samples.d = d; s.t = t;
    return s;
}

}  // namespace test_helpers
