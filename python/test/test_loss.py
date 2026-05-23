"""
Loss and regularizer surface.

- Loss="L2"/"L1"/"huber" and custom callables exercise gdtw.utils.process_function.
- R_cum="huber" / R_inst="huber" exercise the C++ branch in set_loss_functional
  (and the huber_delta argument on the gdtwcpp.solve ABI).

Convergence of L1/L2 on a real warp pair lives in test_warp.py; this file
asserts the loss-function contract independently of any specific warp.
"""
import numpy as np
import pytest

import gdtw

from _helpers import scalar_pair, multid_pair, phi_true


def test_huber_loss_runs():
    t, x, y = scalar_pair(T=200)
    phi, _, f_tau, _ = gdtw.warp(x, y, params={"Loss": "huber"})
    assert np.isfinite(f_tau)
    assert np.max(np.abs(phi(t) - phi_true(t))) < 0.06


def test_huber_matches_half_l2_in_small_residual_regime():
    """For |r| <= delta, huber(r) == 0.5 * r^2; with a huge delta, huber == 0.5 * L2."""
    from gdtw.utils import process_function
    f_huber = process_function("huber", huber_delta=1e9)
    f_l2 = process_function("L2")
    r = np.linspace(-1, 1, 17)
    np.testing.assert_allclose(f_huber(r), 0.5 * f_l2(r), atol=1e-12)


def test_huber_robust_to_outliers():
    """Huber should give a smaller phi-recovery error than L2 when y has outlier spikes."""
    rng = np.random.default_rng(0)
    t, x, y = scalar_pair(T=200)
    spike_idx = rng.choice(len(y), size=5, replace=False)
    y_corr = y.copy()
    y_corr[spike_idx] += 5.0 * rng.standard_normal(len(spike_idx))

    phi_l2, _, _, _ = gdtw.warp(x, y_corr, params={"Loss": "L2"})
    phi_hu, _, _, _ = gdtw.warp(x, y_corr, params={"Loss": "huber", "huber_delta": 0.5})

    err_l2 = np.max(np.abs(phi_l2(t) - phi_true(t)))
    err_hu = np.max(np.abs(phi_hu(t) - phi_true(t)))
    assert err_hu <= err_l2 + 1e-9, f"huber err={err_hu} vs L2 err={err_l2}"


def test_huber_regularizer_cpp_branch():
    """Passing 'huber' as R_cum/R_inst routes through the C++ set_loss_functional branch."""
    t, x, y = scalar_pair(T=200)
    _, _, f_tau, g = gdtw.warp(
        x, y,
        params={"R_cum": "huber", "R_inst": "huber", "huber_delta": 0.5},
    )
    assert np.isfinite(f_tau)
    assert np.max(np.abs(g.phi(t) - phi_true(t))) < 0.1


def test_unknown_loss_string_raises():
    from gdtw.utils import process_function
    with pytest.raises(ValueError, match="huber"):
        process_function("HUBER")


def test_custom_callable_loss_matches_l2_string():
    """User-supplied elementwise loss should match the built-in 'L2' identifier."""
    _, x, y, _ = multid_pair(d=2, T=300, freqs=[5.0, 3.0])
    _, x_tau_str, f_str, _ = gdtw.warp(x, y, params={"Loss": "L2"})
    _, x_tau_fn, f_fn, _ = gdtw.warp(x, y, params={"Loss": lambda r: r ** 2})
    assert f_str == pytest.approx(f_fn, rel=1e-9, abs=1e-12)
    np.testing.assert_allclose(x_tau_str, x_tau_fn, atol=1e-9)
