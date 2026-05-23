/* SPDX-License-Identifier: Apache-2.0
 *
 * Symmetric warping path. Mirrors python/test/test_symmetric.py.
 *
 * Asymmetric mode reads y at t_i. Symmetric mode reads y at
 * psi(t_i, j) = 2*t_i - Tau[i,j] so the residual couples both directions.
 */
#include "doctest/doctest.h"

#include "gdtw/warp.hpp"
#include "_helpers.hpp"

#include <cmath>
#include <vector>

using namespace gdtw;
using namespace test_helpers;

TEST_CASE("test_symmetric_exposes_psi_accessors") {
    auto p = scalar_pair(200);
    GDTWParams params; params.symmetric = true;
    auto r = warp(spec_from(p.x, 1), spec_from(p.y, 1), p.t, params);
    std::vector<double> grid = linspace(0.0, 1.0, 50);
    for (double g : grid) {
        CHECK(std::abs(r.g->psi(g) - (2.0 * g - r.g->phi(g))) < 1e-12);
    }
    CHECK(static_cast<int>(r.g->get_psi_values().size()) == r.g->N());
}

TEST_CASE("test_symmetric_psi_complementary") {
    auto p = scalar_pair(200);
    GDTWParams params; params.symmetric = true;
    auto r = warp(spec_from(p.x, 1), spec_from(p.y, 1), p.t, params);
    auto psi_vals = r.g->get_psi_values();
    for (int i = 0; i < r.g->N(); ++i) {
        CHECK(std::abs(psi_vals[i] + r.g->tau()[i] - 2.0 * r.g->t()[i]) < 1e-12);
    }
}

TEST_CASE("test_symmetric_default_is_asymmetric") {
    auto p = scalar_pair(200);
    auto r_def = warp(spec_from(p.x, 1), spec_from(p.y, 1), p.t);
    GDTWParams params; params.symmetric = false;
    auto r_exp = warp(spec_from(p.x, 1), spec_from(p.y, 1), p.t, params);
    CHECK(r_def.f_tau == doctest::Approx(r_exp.f_tau).epsilon(1e-12));
}
