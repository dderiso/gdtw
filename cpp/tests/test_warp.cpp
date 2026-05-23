/* SPDX-License-Identifier: Apache-2.0
 *
 * End-to-end behavior of gdtw::warp across scalar (d=1) and multi-channel
 * (d>=2) signals. One-for-one port of python/test/test_warp.py.
 */
#include "doctest/doctest.h"

#include "gdtw/warp.hpp"
#include "_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace gdtw;
using namespace test_helpers;

TEST_CASE("test_warp_runs_and_returns_correct_shapes") {
    for (std::size_t d : {std::size_t{1}, std::size_t{2}, std::size_t{3}}) {
        CAPTURE(d);
        auto p = make_pair(d, 300);
        auto r = warp(spec_from(p.x, d), spec_from(p.y, d), p.t);
        const auto& g = *r.g;
        CHECK(static_cast<int>(g.tau().size()) == g.N());
        CHECK(static_cast<int>(g.D().size()) == g.N() * g.M());
        CHECK(r.x_tau.size() == static_cast<std::size_t>(g.N()) * d);
        CHECK(std::isfinite(r.f_tau));
        CHECK(g.d() == d);
    }
}

TEST_CASE("test_warp_recovers_quadratic_phi") {
    struct Row { std::size_t d; Penalty loss; double tol; int T; std::vector<double> freqs; };
    std::vector<Row> rows = {
        {1, Penalty::L2, 0.05, 200, {}},
        {2, Penalty::L2, 0.05, 400, {5.0, 3.0}},
        {2, Penalty::L1, 0.08, 400, {5.0, 3.0}},
    };
    for (const auto& row : rows) {
        CAPTURE(row.d); CAPTURE(static_cast<int>(row.loss)); CAPTURE(row.tol);
        auto p = make_pair(row.d, row.T, row.freqs);
        GDTWParams params; params.loss = row.loss;
        auto r = warp(spec_from(p.x, row.d), spec_from(p.y, row.d), p.t, params);
        CHECK(std::isfinite(r.f_tau));
        CHECK(linf_phi_error(r.phi, p.t) < row.tol);
    }
}

TEST_CASE("test_warp_xtau_matches_y_2d") {
    auto p = make_pair(2, 400, {5.0, 3.0});
    auto r = warp(spec_from(p.x, 2), spec_from(p.y, 2), p.t);
    const auto& g = *r.g;
    REQUIRE(r.x_tau.size() == g.signal_y().samples().size());
    double mx = 0.0;
    for (std::size_t i = 0; i < r.x_tau.size(); ++i) {
        double e = std::abs(r.x_tau[i] - g.signal_y().samples()[i]);
        if (e > mx) mx = e;
    }
    CHECK(mx < 0.3);
}

TEST_CASE("test_warp_identity_when_x_equals_y") {
    struct Row { std::size_t d; bool symmetric; };
    std::vector<Row> rows = { {1, false}, {3, false}, {1, true} };
    for (const auto& row : rows) {
        CAPTURE(row.d); CAPTURE(row.symmetric);
        int T = (row.d == 1) ? 200 : 300;
        std::vector<double> t = linspace(0.0, 1.0, T);
        std::vector<double> x;
        if (row.d == 1) {
            x.resize(T);
            for (int i = 0; i < T; ++i) x[i] = std::sin(2.0 * M_PI * 5.0 * t[i]);
        } else {
            std::mt19937_64 rng(0);
            std::normal_distribution<double> n(0.0, 1.0);
            x.assign(T * row.d, 0.0);
            for (std::size_t k = 0; k < row.d; ++k) {
                double cum = 0.0;
                for (int i = 0; i < T; ++i) {
                    cum += n(rng);
                    x[i * row.d + k] = cum;
                }
            }
        }
        std::vector<double> y = x;

        GDTWParams params; params.symmetric = row.symmetric;
        auto r = warp(spec_from(x, row.d), spec_from(y, row.d), t, params);
        double mx = 0.0;
        for (double ti : t) {
            double e = std::abs(r.phi(ti) - ti);
            if (e > mx) mx = e;
        }
        CHECK(mx < 5e-2);

        /* Perturb: roll y by 5 along the leading (time) axis. */
        std::vector<double> y_roll = y;
        std::rotate(y_roll.rbegin(), y_roll.rbegin() + 5 * static_cast<int>(row.d), y_roll.rend());
        GDTWParams params2; params2.symmetric = row.symmetric;
        auto r2 = warp(spec_from(x, row.d), spec_from(y_roll, row.d), t, params2);
        CHECK(r.f_tau < r2.f_tau);
    }
}

TEST_CASE("test_warp_callable_input_routing") {
    for (std::size_t d : {std::size_t{1}, std::size_t{2}}) {
        CAPTURE(d);
        std::vector<double> t = linspace(0.0, 1.0, 300);
        SignalSpec x, y;
        if (d == 1) {
            x = SignalCallableScalar{[](double u){ return std::sin(2.0 * M_PI * 5.0 * u); }};
            y = SignalCallableScalar{[](double u){ return std::sin(2.0 * M_PI * 5.0 * u * u); }};
        } else {
            x = SignalCallableVector{[](double u){
                return std::vector<double>{
                    std::sin(2.0 * M_PI * 5.0 * u),
                    std::cos(2.0 * M_PI * 3.0 * u)
                };
            }};
            y = SignalCallableVector{[](double u){
                double u2 = u * u;
                return std::vector<double>{
                    std::sin(2.0 * M_PI * 5.0 * u2),
                    std::cos(2.0 * M_PI * 3.0 * u2)
                };
            }};
        }
        auto r = warp(x, y, t);
        CHECK(linf_phi_error(r.phi, t) < 0.05);
    }
}

TEST_CASE("test_warp_tuple_input_routing") {
    for (std::size_t d : {std::size_t{1}, std::size_t{2}}) {
        CAPTURE(d);
        std::vector<double> t;
        std::vector<double> x, y;
        if (d == 1) {
            Pair1D s = scalar_pair(200);
            t = s.t; x = s.x; y = s.y;
        } else {
            PairMD s = multid_pair(2, 200, {5.0, 3.0});
            t = s.t; x = s.x; y = s.y;
        }
        auto r_arr   = warp(spec_from(x, d), spec_from(y, d), t);
        auto r_tuple = warp(spec_from_with_t(x, d, t), spec_from_with_t(y, d, t), t);
        std::vector<double> grid = linspace(0.0, 1.0, 50);
        for (double g : grid) {
            CHECK(std::abs(r_arr.phi(g) - r_tuple.phi(g)) < 1e-9);
        }
    }
}

TEST_CASE("test_warp_mismatched_channel_dim_raises") {
    int T = 200;
    std::vector<double> t = linspace(0.0, 1.0, T);
    std::vector<double> x(T * 2), y(T * 3);
    for (int i = 0; i < T; ++i) {
        x[i * 2 + 0] = std::sin(t[i]);   x[i * 2 + 1] = std::cos(t[i]);
        y[i * 3 + 0] = std::sin(t[i]);   y[i * 3 + 1] = std::cos(t[i]);   y[i * 3 + 2] = std::sin(2.0 * t[i]);
    }
    CHECK_THROWS_AS(warp(spec_from(x, 2), spec_from(y, 3), t), GDTWError);
}

TEST_CASE("test_warp_baseline_pinned") {
    /* Numerical regression anchor: must match python/test/test_warp.py::test_warp_baseline_pinned. */
    auto p = scalar_pair(200);
    auto r = warp(spec_from(p.x, 1), spec_from(p.y, 1), p.t);
    CHECK(r.f_tau == doctest::Approx(0.16694595777903623).epsilon(1e-6));
    CHECK(r.g->phi(0.5) == doctest::Approx(0.25318182065).epsilon(1e-6));
}
