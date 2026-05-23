/* SPDX-License-Identifier: Apache-2.0
 *
 * Signal class + scale() utility. Mirrors python/test/test_signal.py.
 */
#include "doctest/doctest.h"

#include "gdtw/signal.hpp"
#include "gdtw/utils.hpp"
#include "_helpers.hpp"

#include <cmath>
#include <limits>
#include <vector>

using namespace gdtw;
using namespace test_helpers;

TEST_CASE("test_signal_per_channel_scaling_independent") {
    int T = 200;
    std::vector<double> t = linspace(0.0, 1.0, T);
    std::vector<double> z(T * 2);
    for (int i = 0; i < T; ++i) {
        z[i * 2 + 0] = 1000.0 * std::sin(2.0 * M_PI * 2.0 * t[i]);
        z[i * 2 + 1] = 0.001  * std::cos(2.0 * M_PI * 3.0 * t[i]);
    }
    SignalSamples s; s.data = z; s.d = 2;
    Signal sig(s, "z", T, /*scale_signals=*/true);
    sig.check_signal();
    const auto& a = sig.samples();
    double mn0 = std::numeric_limits<double>::infinity(), mx0 = -mn0;
    double mn1 = std::numeric_limits<double>::infinity(), mx1 = -mn1;
    for (int i = 0; i < T; ++i) {
        mn0 = std::min(mn0, a[i * 2 + 0]); mx0 = std::max(mx0, a[i * 2 + 0]);
        mn1 = std::min(mn1, a[i * 2 + 1]); mx1 = std::max(mx1, a[i * 2 + 1]);
    }
    CHECK(mn0 == doctest::Approx(-1.0).epsilon(1e-10));
    CHECK(mx0 == doctest::Approx( 1.0).epsilon(1e-10));
    CHECK(mn1 == doctest::Approx(-1.0).epsilon(1e-10));
    CHECK(mx1 == doctest::Approx( 1.0).epsilon(1e-10));
}

TEST_CASE("test_scale_util_per_channel") {
    /* [[0,100],[10,200],[20,300]] row-major, d=2. */
    std::vector<double> z = {0.0, 100.0, 10.0, 200.0, 20.0, 300.0};
    auto out = scale(z, /*d=*/2, -1.0, 1.0);
    /* Per-column min/max should hit [-1, 1] exactly. */
    double mn0 = std::numeric_limits<double>::infinity(), mx0 = -mn0;
    double mn1 = std::numeric_limits<double>::infinity(), mx1 = -mn1;
    for (int i = 0; i < 3; ++i) {
        mn0 = std::min(mn0, out[i * 2 + 0]); mx0 = std::max(mx0, out[i * 2 + 0]);
        mn1 = std::min(mn1, out[i * 2 + 1]); mx1 = std::max(mx1, out[i * 2 + 1]);
    }
    CHECK(mn0 == doctest::Approx(-1.0));
    CHECK(mx0 == doctest::Approx( 1.0));
    CHECK(mn1 == doctest::Approx(-1.0));
    CHECK(mx1 == doctest::Approx( 1.0));
}
