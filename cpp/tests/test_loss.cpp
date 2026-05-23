/* SPDX-License-Identifier: Apache-2.0
 *
 * Loss and regularizer surface. Mirrors python/test/test_loss.py.
 */
#include "doctest/doctest.h"

#include "gdtw/warp.hpp"
#include "gdtw/utils.hpp"
#include "_helpers.hpp"

#include <cmath>
#include <random>
#include <vector>

using namespace gdtw;
using namespace test_helpers;

TEST_CASE("test_huber_loss_runs") {
    auto p = scalar_pair(200);
    GDTWParams params; params.loss = Penalty::HUBER;
    auto r = warp(spec_from(p.x, 1), spec_from(p.y, 1), p.t, params);
    CHECK(std::isfinite(r.f_tau));
    double mx = 0.0;
    for (double ti : p.t) {
        double e = std::abs(r.phi(ti) - phi_true(ti));
        if (e > mx) mx = e;
    }
    CHECK(mx < 0.06);
}

TEST_CASE("test_huber_matches_half_l2_in_small_residual_regime") {
    auto f_huber = make_penalty(Penalty::HUBER, 1e9);
    auto f_l2    = make_penalty(Penalty::L2);
    std::vector<double> r = linspace(-1.0, 1.0, 17);
    for (double v : r) {
        CHECK(f_huber(v) == doctest::Approx(0.5 * f_l2(v)).epsilon(1e-12));
    }
}

TEST_CASE("test_huber_robust_to_outliers") {
    std::mt19937_64 rng(0);
    auto p = scalar_pair(200);
    std::vector<double> y_corr = p.y;
    /* Match numpy.random.default_rng(0).choice(len(y), size=5, replace=False) intent:
     * pick 5 random distinct indices and spike them. The exact indices need not
     * match Python's PRNG — the property under test is that huber beats L2 on a
     * sample with several large outliers.
     */
    std::vector<int> idx;
    {
        std::vector<int> all(p.y.size());
        for (std::size_t i = 0; i < all.size(); ++i) all[i] = static_cast<int>(i);
        std::shuffle(all.begin(), all.end(), rng);
        for (int i = 0; i < 5; ++i) idx.push_back(all[i]);
    }
    std::normal_distribution<double> n(0.0, 1.0);
    for (int j : idx) y_corr[j] += 5.0 * n(rng);

    GDTWParams pl; pl.loss = Penalty::L2;
    GDTWParams ph; ph.loss = Penalty::HUBER; ph.huber_delta = 0.5;
    auto r_l2 = warp(spec_from(p.x, 1), spec_from(y_corr, 1), p.t, pl);
    auto r_hu = warp(spec_from(p.x, 1), spec_from(y_corr, 1), p.t, ph);

    double err_l2 = 0.0, err_hu = 0.0;
    for (double ti : p.t) {
        err_l2 = std::max(err_l2, std::abs(r_l2.phi(ti) - phi_true(ti)));
        err_hu = std::max(err_hu, std::abs(r_hu.phi(ti) - phi_true(ti)));
    }
    CHECK(err_hu <= err_l2 + 1e-9);
}

TEST_CASE("test_huber_regularizer_cpp_branch") {
    auto p = scalar_pair(200);
    GDTWParams params;
    params.r_cum = Penalty::HUBER;
    params.r_inst = Penalty::HUBER;
    params.huber_delta = 0.5;
    auto r = warp(spec_from(p.x, 1), spec_from(p.y, 1), p.t, params);
    CHECK(std::isfinite(r.f_tau));
    double mx = 0.0;
    for (double ti : p.t) {
        double e = std::abs(r.g->phi(ti) - phi_true(ti));
        if (e > mx) mx = e;
    }
    CHECK(mx < 0.1);
}

TEST_CASE("test_custom_callable_loss_matches_l2_string") {
    /* In C++ the user supplies a PenaltyFn directly with loss=Penalty::CUSTOM. */
    auto p = multid_pair(2, 300, {5.0, 3.0});

    GDTWParams ps; ps.loss = Penalty::L2;
    GDTWParams pf; pf.loss = Penalty::CUSTOM;
    pf.custom_loss = [](double r){ return r * r; };

    auto r_str = warp(spec_from(p.x, 2), spec_from(p.y, 2), p.t, ps);
    auto r_fn  = warp(spec_from(p.x, 2), spec_from(p.y, 2), p.t, pf);

    CHECK(r_str.f_tau == doctest::Approx(r_fn.f_tau).epsilon(1e-9));
    REQUIRE(r_str.x_tau.size() == r_fn.x_tau.size());
    for (std::size_t i = 0; i < r_str.x_tau.size(); ++i) {
        CHECK(std::abs(r_str.x_tau[i] - r_fn.x_tau[i]) < 1e-9);
    }
}
