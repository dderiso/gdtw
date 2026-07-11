# SPDX-License-Identifier: Apache-2.0
#
# Pins the solver's numerical conventions so drift is caught:
#   - the objective quadrature (pinned mode: w_0 = 1 initial node, kept
#     bit-compatible with published releases; relaxed mode: w_0 = dt_0),
#   - the extended relaxed-boundary bounds (the paper's formula: beta
#     relaxes the two s_max cones only),
#   - the instantaneous penalty applied to the raw discrete slope.

import numpy as np
import pytest

from gdtw.gdtw import GDTW


def _fit(x, y, T=80, **overrides):
    t = np.linspace(0.0, 1.0, T)
    params = dict(x=x, y=y, t=t, lambda_cum=0.4, lambda_inst=0.15,
                  s_min=0.2, s_max=5.0, Loss="L2", R_cum="L2", R_inst="L2",
                  max_iters=4, verbose=0)
    params.update(overrides)
    g = GDTW().set_params(params)
    g.run()
    return g


def _objective_from_tau(g, w0):
    """Recompute the documented objective at the returned warp values:
    w0 * n(0) + sum_i dt_i * (lambda_inst * R_inst(slope_i) + n(i+1)),
    with n(i) = L(x(tau_i) - y(t_i)) + lambda_cum * R_cum(tau_i - t_i)
    and R_inst applied to the raw slope."""
    t, tau = g.t, g.tau
    x_tau = np.asarray(g.x_f(tau), dtype=float)
    y_t = np.asarray(g.y_f(t), dtype=float)
    resid = x_tau - y_t
    if resid.ndim == 2:
        n = (resid ** 2).sum(axis=1)
    else:
        n = resid ** 2
    n = n + g.lambda_cum * (tau - t) ** 2
    dt = np.diff(t)
    slopes = np.diff(tau) / dt
    e = g.lambda_inst * slopes ** 2
    return w0 * n[0] + float(np.sum(dt * (e + n[1:])))


def _bump_pair(T=80, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, T)
    y = np.exp(-((t - 0.45) / 0.1) ** 2)
    x = np.interp(t + 0.07 * np.sin(np.pi * t), t, y) + 0.02 * rng.standard_normal(T)
    return x, y


def test_value_convention_pinned_endpoints():
    x, y = _bump_pair()
    g = _fit(x, y)
    assert g.f_tau == pytest.approx(_objective_from_tau(g, w0=1.0), rel=1e-9)


def test_value_convention_relaxed_boundaries():
    x, y = _bump_pair(seed=1)
    g = _fit(x, y, BC_start_stop=False, s_beta=0.1)
    dt0 = float(g.t[1] - g.t[0])
    assert g.f_tau == pytest.approx(_objective_from_tau(g, w0=dt0), rel=1e-9)


def test_pinned_and_relaxed_agree_up_to_boundary_constant():
    # with beta = 0 the relaxed solve explores the same interior; its
    # optimal warp evaluated under the pinned convention can only match
    # or improve on the pinned optimum up to the boundary reweighting
    x, y = _bump_pair(seed=2)
    g_pin = _fit(x, y)
    g_rel = _fit(x, y, BC_start_stop=False, s_beta=0.0)
    # both must satisfy their own documented conventions (checked above);
    # here: the two warps agree closely away from the boundary
    interior = slice(2, -2)
    assert np.max(np.abs(g_pin.tau[interior] - g_rel.tau[interior])) < 0.05


def test_beta_bounds_match_paper_formula():
    x, y = _bump_pair(seed=3)
    for beta in (0.0, 0.15):
        g = GDTW().set_params(dict(x=x, y=y, t=np.linspace(0, 1, 80),
                                   s_min=0.3, s_max=3.0, s_beta=beta,
                                   BC_start_stop=(beta == 0.0), verbose=0))
        g.check_params()
        g.allocate()
        g.compute_taus()
        t = g.t
        u_expect = np.clip(np.minimum(3.0 * t + beta, 1 - 0.3 * (1 - t)), 0.0, 1.0)
        l_expect = np.clip(np.maximum(0.3 * t, 1 - 3.0 * (1 - t) - beta), 0.0, 1.0)
        assert np.allclose(g.u, u_expect, atol=1e-12)
        assert np.allclose(g.l, l_expect, atol=1e-12)
        # beta relaxes only the boundary rows' values, inside [0, 1]
        assert g.u[0] == pytest.approx(min(beta, 1.0))
        assert g.l[-1] == pytest.approx(max(1.0 - beta, 0.0))
