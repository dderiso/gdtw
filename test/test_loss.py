"""
Huber loss support.

- Loss="huber" exercises the Python-side branch in utils.process_function.
- R_cuml="huber" / R_inst="huber" exercise the C++ branch in set_loss_functional
  (and the new huber_delta argument on the gdtwcpp.solve ABI).
"""
import numpy as np
import pytest

import gdtw


def _scalar_pair(T=200, freq=5.0):
    t = np.linspace(0, 1, T)
    phi_true = t ** 2
    x = np.sin(2 * np.pi * freq * t)
    y = np.sin(2 * np.pi * freq * phi_true)
    return t, x, y


def test_huber_loss_runs():
    t, x, y = _scalar_pair(T=200)
    phi, _, f_tau, _ = gdtw.warp(x, y, params={"Loss": "huber"})
    assert np.isfinite(f_tau)
    # Same DP-grid tolerance as L2 on a clean signal.
    assert np.max(np.abs(phi(t) - t ** 2)) < 0.06


def test_huber_loss_matches_half_l2_in_small_residual_regime():
    """For |r| <= delta, huber(r) == 0.5 * r^2; with a huge delta, huber == 0.5 * L2."""
    from gdtw.utils import process_function
    f_huber = process_function("huber", huber_delta=1e9)
    f_l2 = process_function("L2")
    r = np.linspace(-1, 1, 17)
    np.testing.assert_allclose(f_huber(r), 0.5 * f_l2(r), atol=1e-12)


def test_huber_loss_robust_to_outliers():
    """Huber should give a smaller phi-recovery error than L2 when y has outlier spikes."""
    rng = np.random.default_rng(0)
    t, x, y = _scalar_pair(T=200)
    # Inject a handful of spikes that swamp L2.
    spike_idx = rng.choice(len(y), size=5, replace=False)
    y_corr = y.copy()
    y_corr[spike_idx] += 5.0 * rng.standard_normal(len(spike_idx))

    phi_l2, _, _, _ = gdtw.warp(x, y_corr, params={"Loss": "L2"})
    phi_hu, _, _, _ = gdtw.warp(x, y_corr, params={"Loss": "huber", "huber_delta": 0.5})

    err_l2 = np.max(np.abs(phi_l2(t) - t ** 2))
    err_hu = np.max(np.abs(phi_hu(t) - t ** 2))
    assert err_hu <= err_l2 + 1e-9, f"huber err={err_hu} vs L2 err={err_l2}"


def test_huber_regularizer_cpp_branch():
    """Passing 'huber' as R_cuml/R_inst routes through the new C++ set_loss_functional branch."""
    t, x, y = _scalar_pair(T=200)
    _, _, f_tau, g = gdtw.warp(
        x, y,
        params={"R_cum": "huber", "R_inst": "huber", "huber_delta": 0.5},
    )
    assert np.isfinite(f_tau)
    # Solver should still recover something close to the true phi.
    assert np.max(np.abs(g.phi(t) - t ** 2)) < 0.1


def test_huber_unknown_string_raises():
    """A typo'd string still raises a clear error (now mentions 'huber')."""
    from gdtw.utils import process_function
    with pytest.raises(ValueError, match="huber"):
        process_function("HUBER")
