"""
Regression tests for the 1-D (scalar) warp path.

These pin the pre-patch baseline so that the multi-D refactor cannot silently drift
scalar results. Baseline values were captured from the current installed gdtw before
introducing per-channel broadcasting in compute_dist_matrix.
"""
import numpy as np
import pytest

import gdtw


def _build_scalar_pair(T=200):
    t = np.linspace(0, 1, T)
    phi_true = t ** 2
    x = np.sin(2 * np.pi * 5 * t)
    y = np.sin(2 * np.pi * 5 * phi_true)
    return t, x, y


def test_scalar_warp_runs_and_returns_correct_shapes():
    t, x, y = _build_scalar_pair(T=200)
    phi, x_tau, f_tau, g = gdtw.warp(x, y)
    assert x_tau.shape == (200,)
    assert g.tau.shape == (200,)
    assert g.D.shape == (g.N, g.M)
    assert np.isfinite(f_tau)


def test_scalar_warp_recovers_quadratic_phi():
    t, x, y = _build_scalar_pair(T=200)
    phi, x_tau, f_tau, g = gdtw.warp(x, y)
    # Recovery should be within DP-grid quantization tolerance.
    assert np.max(np.abs(phi(t) - t ** 2)) < 0.05


def test_scalar_warp_matches_baseline():
    """Baseline pinned from the pre-multi-D code (see plan)."""
    t, x, y = _build_scalar_pair(T=200)
    _, _, f_tau, g = gdtw.warp(x, y)
    # f_tau should match the pre-patch value to ~6 decimals on this fixed grid.
    assert f_tau == pytest.approx(0.16694595777903623, rel=1e-6, abs=1e-8)
    # phi(0.5) baseline.
    assert float(g.phi(0.5)) == pytest.approx(0.25318182065, rel=1e-6, abs=1e-8)


def test_callable_signal_path():
    """Functional signals route through the callable branch of Signal.check_signal."""
    t = np.linspace(0, 1, 300)
    x_fn = lambda u: np.sin(2 * np.pi * 5 * u)
    y_fn = lambda u: np.sin(2 * np.pi * 5 * u ** 2)
    phi, x_tau, f_tau, g = gdtw.warp(x_fn, y_fn, t=t)
    assert np.max(np.abs(phi(t) - t ** 2)) < 0.05


def test_tuple_signal_path():
    """Tuple form (array, t) routes through the tuple branch."""
    t = np.linspace(0, 1, 200)
    x = np.sin(2 * np.pi * 5 * t)
    y = np.sin(2 * np.pi * 5 * t ** 2)
    phi, _, _, g = gdtw.warp((x, t), (y, t))
    assert np.max(np.abs(phi(t) - t ** 2)) < 0.05
