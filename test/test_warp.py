"""
End-to-end behavior of gdtw.warp() across scalar (d=1) and multi-channel (d>=2)
signals. Tests are parametrized over d so each property is asserted once per
relevant dimensionality instead of being duplicated as scalar/multi-D twins.
"""
import numpy as np
import pytest

import gdtw

from _helpers import make_pair, multid_pair, linf_phi_error, phi_true


@pytest.mark.parametrize("d", [1, 2, 3])
def test_warp_runs_and_returns_correct_shapes(d):
    t, x, y = make_pair(d=d, T=300)
    phi, x_tau, f_tau, g = gdtw.warp(x, y)
    assert g.tau.shape == (g.N,)
    assert g.D.shape == (g.N, g.M)
    assert x_tau.shape == ((g.N,) if d == 1 else (g.N, d))
    assert np.isfinite(f_tau)
    assert g.d == d


@pytest.mark.parametrize("d, loss, tol", [
    (1, "L2", 0.05),
    (2, "L2", 0.05),
    (2, "L1", 0.08),
])
def test_warp_recovers_quadratic_phi(d, loss, tol):
    if d == 1:
        t, x, y = make_pair(d=1, T=200)
    else:
        t, x, y = make_pair(d=d, T=400, freqs=[5.0, 3.0])
    phi, _, f_tau, _ = gdtw.warp(x, y, params={"Loss": loss})
    assert np.isfinite(f_tau)
    assert linf_phi_error(phi, t) < tol


def test_warp_xtau_matches_y_2d():
    """x(phi(t)) should match y(t) within DP-grid tolerance, per channel."""
    t, x, y = make_pair(d=2, T=400, freqs=[5.0, 3.0])
    _, x_tau, _, g = gdtw.warp(x, y)
    assert x_tau.shape == g.y_a.shape
    assert np.max(np.abs(x_tau - g.y_a)) < 0.3


@pytest.mark.parametrize("d, symmetric", [
    (1, False),
    (3, False),
    (1, True),
])
def test_warp_identity_when_x_equals_y(d, symmetric):
    """When x == y, phi must collapse to the identity (both modes)."""
    T = 200 if d == 1 else 300
    if d == 1:
        t = np.linspace(0, 1, T)
        x = np.sin(2 * np.pi * 5 * t)
        y = x.copy()
    else:
        rng = np.random.default_rng(0)
        t = np.linspace(0, 1, T)
        x = np.cumsum(rng.standard_normal((T, d)), axis=0)
        y = x.copy()
    phi, _, f_tau, _ = gdtw.warp(x, y, params={"symmetric": symmetric})
    assert np.max(np.abs(phi(t) - t)) < 5e-2
    # Identity warp should beat a perturbed warp on the same pair.
    _, _, f_perturbed, _ = gdtw.warp(x, np.roll(y, 5, axis=0), params={"symmetric": symmetric})
    assert f_tau < f_perturbed


@pytest.mark.parametrize("d", [1, 2])
def test_warp_callable_input_routing(d):
    t = np.linspace(0, 1, 300)
    if d == 1:
        x_fn = lambda u: np.sin(2 * np.pi * 5 * u)
        y_fn = lambda u: np.sin(2 * np.pi * 5 * u ** 2)
    else:
        x_fn = lambda u: np.column_stack([np.sin(2 * np.pi * 5 * u), np.cos(2 * np.pi * 3 * u)])
        y_fn = lambda u: np.column_stack([np.sin(2 * np.pi * 5 * u ** 2), np.cos(2 * np.pi * 3 * u ** 2)])
    phi, _, _, _ = gdtw.warp(x_fn, y_fn, t=t)
    assert linf_phi_error(phi, t) < 0.05


@pytest.mark.parametrize("d", [1, 2])
def test_warp_tuple_input_routing(d):
    """Tuple form (array, t) must agree with the array form on the same grid."""
    t = np.linspace(0, 1, 200)
    if d == 1:
        _, x, y = make_pair(d=1, T=200)
    else:
        t, x, y = make_pair(d=2, T=200, freqs=[5.0, 3.0])
    phi_arr, _, _, _ = gdtw.warp(x, y, t=t)
    phi_tuple, _, _, _ = gdtw.warp((x, t), (y, t), t=t)
    grid = np.linspace(0, 1, 50)
    np.testing.assert_allclose(phi_arr(grid), phi_tuple(grid), atol=1e-9)


def test_warp_mismatched_channel_dim_raises():
    T = 200
    t = np.linspace(0, 1, T)
    x = np.column_stack([np.sin(t), np.cos(t)])               # d=2
    y = np.column_stack([np.sin(t), np.cos(t), np.sin(2 * t)])  # d=3
    with pytest.raises(ValueError, match=r"d=2.*d=3|d=3.*d=2"):
        gdtw.warp(x, y)


def test_warp_higher_rank_input_raises():
    T = 100
    bad = np.zeros((T, 2, 2))
    good = np.zeros((T, 2))
    with pytest.raises(ValueError, match=r"1-D \(T,\) or 2-D \(T, d\)"):
        gdtw.warp(bad, good)


def test_warp_baseline_pinned():
    """Numerical regression anchor — pinned from the pre-multi-D code path."""
    t, x, y = make_pair(d=1, T=200)
    _, _, f_tau, g = gdtw.warp(x, y)
    assert f_tau == pytest.approx(0.16694595777903623, rel=1e-6, abs=1e-8)
    assert float(g.phi(0.5)) == pytest.approx(0.25318182065, rel=1e-6, abs=1e-8)
