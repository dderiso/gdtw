"""
Symmetric warping path: ψ(t) = 2t − φ(t) computed inside the same DP solve.

Asymmetric mode reads y at t_i and varies x at Tau[i, j]. Symmetric mode keeps the
same DP, but reads y at ψ(t_i, j) = 2 t_i − Tau[i, j] so the residual couples both
directions. The DP itself is unchanged; only the distance matrix and the exposed
accessors differ.
"""
import numpy as np
import pytest

import gdtw


def _scalar_pair(T=200):
    t = np.linspace(0, 1, T)
    phi_true = t ** 2
    x = np.sin(2 * np.pi * 5 * t)
    y = np.sin(2 * np.pi * 5 * phi_true)
    return t, x, y


def test_symmetric_runs_and_returns_correct_shapes():
    _, x, y = _scalar_pair(T=200)
    phi, x_tau, f_tau, g = gdtw.warp(x, y, params={"symmetric": True})
    assert g.tau.shape == (g.N,)
    assert g.D.shape == (g.N, g.M)
    assert np.isfinite(f_tau)
    # Both accessors exist and are consistent with phi.
    grid = np.linspace(0, 1, 50)
    np.testing.assert_allclose(g.psi(grid), 2.0 * grid - g.phi(grid), atol=1e-12)
    assert g.get_psi_values().shape == (g.N,)


def test_symmetric_psi_complementary():
    """ψ(t) + φ(t) == 2t at every node, by construction."""
    _, x, y = _scalar_pair(T=200)
    _, _, _, g = gdtw.warp(x, y, params={"symmetric": True})
    np.testing.assert_allclose(g.get_psi_values() + g.tau, 2.0 * g.t, atol=1e-12)


def test_symmetric_identity_when_x_equals_y():
    """When x == y there is no warping to do; symmetric mode must collapse to φ = identity.

    (Symmetric mode optimizes a different objective than asymmetric — Loss(x(φ) − y(2t−φ)) —
    so we cannot use the asymmetric ground-truth recovery test here. The identity-warp case
    is the meaningful invariant.)
    """
    T = 200
    t = np.linspace(0, 1, T)
    x = np.sin(2 * np.pi * 5 * t)
    phi, _, _, _ = gdtw.warp(x, x.copy(), params={"symmetric": True})
    assert np.max(np.abs(phi(t) - t)) < 0.05


def test_symmetric_default_is_asymmetric():
    """Omitting `symmetric` must reproduce the legacy asymmetric path bit-identically."""
    t, x, y = _scalar_pair(T=200)
    _, _, f_default, _ = gdtw.warp(x, y)
    _, _, f_explicit, _ = gdtw.warp(x, y, params={"symmetric": False})
    assert f_default == pytest.approx(f_explicit, rel=1e-12, abs=1e-12)
