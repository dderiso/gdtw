# SPDX-License-Identifier: Apache-2.0
#
# The C++ kernel against an independent NumPy reference DP (matrix-style, no
# shared code path): identical optimal warps, values to machine precision, in
# both boundary modes and across penalties -- the guard that the two-pointer
# feasible-window optimization and the rolling-row refactor are exact. Plus
# the robustness surface: callable regularizers (including ones that raise),
# GIL-released threading, best-pass refinement, and the constant-channel
# scale guard.

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import gdtw
from gdtw.gdtw import GDTW
from gdtwcpp import solve as kernel_solve

from _helpers import scalar_pair

_PEN = {
    "L2": lambda u, d: u * u,
    "L1": lambda u, d: np.abs(u),
    "huber": lambda u, d: np.where(np.abs(u) <= d, 0.5 * u * u, d * (np.abs(u) - 0.5 * d)),
}


def _prepared(T=60, seed=0, BC=True, **overrides):
    """A GDTW object with graph and costs built, ready for one kernel call."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, T)
    y = np.exp(-((t - 0.45) / 0.1) ** 2)
    x = np.interp(t + 0.07 * np.sin(np.pi * t), t, y) + 0.02 * rng.standard_normal(T)
    params = dict(x=x, y=y, t=t, lambda_cum=0.4, lambda_inst=0.15,
                  s_min=0.2, s_max=5.0, Loss="L2", R_cum="L2", R_inst="L2",
                  BC_start_stop=BC, verbose=0)
    params.update(overrides)
    g = GDTW().set_params(params)
    g.check_params()
    g.allocate()
    g.compute_taus()
    g.compute_dist_matrix()
    return g


def _kernel(g):
    i = kernel_solve(g.t, g.Tau, g.D, g.R_cum, g.R_inst,
                     np.double(g.lambda_cum), np.double(g.lambda_inst),
                     np.double(g.s_min), np.double(g.s_max),
                     np.double(g.huber_delta), g.BC_start_stop, 0,
                     g.tau, g.path, g.f_tau)
    assert i == 1
    return float(g.f_tau), g.tau.copy()


def _oracle(g):
    """Reference DP under the documented convention: node costs
    n = D + lambda_cum R_cum(Tau - t), start weight w0 = 1 (pinned) or dt_0
    (relaxed), transitions f + dt (lambda_inst R_inst(raw slope) + n(i+1))."""
    t, Tau, D = g.t, g.Tau, g.D
    N, M = Tau.shape
    pen_c, pen_i = _PEN[g.R_cum], _PEN[g.R_inst]
    d = g.huber_delta
    node = D + g.lambda_cum * pen_c(Tau - t[:, None], d)
    f = np.full((N, M), np.inf)
    back = np.zeros((N, M), dtype=int)
    jc = (M - 1) // 2
    if g.BC_start_stop:
        f[0, jc] = node[0, jc]
    else:
        f[0] = (t[1] - t[0]) * node[0]
    for i in range(N - 1):
        dt = t[i + 1] - t[i]
        slopes = (Tau[i + 1][None, :] - Tau[i][:, None]) / dt
        feas = (slopes >= g.s_min - 1e-10) & (slopes <= g.s_max + 1e-10)
        edge = g.lambda_inst * pen_i(slopes, d)
        total = f[i][:, None] + np.where(feas, dt * (edge + node[i + 1][None, :]), np.inf)
        back[i + 1] = np.argmin(total, axis=0)
        f[i + 1] = total[back[i + 1], np.arange(M)]
    j = jc if g.BC_start_stop else int(np.argmin(f[N - 1]))
    value = float(f[N - 1, j])
    tau = np.zeros(N)
    for i in range(N - 1, -1, -1):
        tau[i] = Tau[i, j]
        if i > 0:
            j = int(back[i, j])
    return value, tau


@pytest.mark.parametrize("BC", [True, False])
@pytest.mark.parametrize("pens", [("L2", "L2"), ("L1", "L2"), ("L2", "huber"), ("huber", "L1")])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kernel_matches_oracle(BC, pens, seed):
    g = _prepared(seed=seed, BC=BC, R_cum=pens[0], R_inst=pens[1],
                  s_beta=(0.15 if not BC else 0.0))
    v_kernel, tau_kernel = _kernel(g)
    v_ref, tau_ref = _oracle(g)
    assert v_kernel == pytest.approx(v_ref, rel=1e-12, abs=1e-15)
    np.testing.assert_array_equal(tau_kernel, tau_ref)


def test_callable_regularizer_matches_string():
    """A Python-callable penalty routes through the std::function path and
    must agree with the built-in code path."""
    t, x, y = scalar_pair(T=120)
    _, _, f_str, g_str = gdtw.warp(x, y, params={"R_cum": "L2", "R_inst": "L2"})
    _, _, f_fn, g_fn = gdtw.warp(x, y, params={"R_cum": lambda u: u * u,
                                               "R_inst": lambda u: u * u})
    assert f_fn == pytest.approx(f_str, rel=1e-9, abs=1e-12)
    np.testing.assert_allclose(g_fn.tau, g_str.tau, atol=1e-12)


def test_raising_callable_is_a_python_error():
    """A penalty that raises must surface as a Python exception, not
    terminate the interpreter."""
    t, x, y = scalar_pair(T=60)

    def bad(u):
        raise RuntimeError("boom")

    with pytest.raises(Exception):
        gdtw.warp(x, y, params={"R_cum": bad, "max_iters": 1})


def test_unknown_penalty_string_raises():
    t, x, y = scalar_pair(T=60)
    with pytest.raises(Exception, match="Unknown string"):
        gdtw.warp(x, y, params={"R_cum": "L3", "max_iters": 1})


def test_threaded_solves_match_sequential():
    """The string-penalty kernel releases the GIL; concurrent solves must
    reproduce the sequential results exactly."""
    pairs = [scalar_pair(T=150)[1:] for _ in range(6)]
    seq = [gdtw.warp(x, y)[2] for x, y in pairs]
    with ThreadPoolExecutor(max_workers=4) as ex:
        par = list(ex.map(lambda p: gdtw.warp(p[0], p[1])[2], pairs))
    np.testing.assert_allclose(par, seq, rtol=0, atol=0)


def test_refinement_keeps_best_pass():
    """f_tau must equal the minimum over refinement passes, and tau must be
    the warp of that pass (a re-grid need not contain the incumbent, so a
    later pass can regress)."""
    t, x, y = scalar_pair(T=200)
    passes = []
    g = GDTW().set_params(dict(x=x, y=y, t=t, max_iters=8,
                               callback=lambda s: passes.append(float(s.f_tau)),
                               verbose=0)).run()
    assert len(passes) >= 2
    assert float(g.f_tau) == min(passes)


def test_constant_channel_scaling_has_no_nans():
    """A constant channel used to divide by zero in utils.scale."""
    T = 100
    t = np.linspace(0, 1, T)
    x = np.column_stack([np.sin(2 * np.pi * 3 * t), np.ones(T)])
    y = np.column_stack([np.sin(2 * np.pi * 3 * t ** 1.2), np.ones(T)])
    phi, x_tau, f_tau, g = gdtw.warp(x, y)
    assert np.all(np.isfinite(x_tau))
    assert np.isfinite(f_tau)
