"""
Multi-dimensional signal warping.

A single warping function phi: [0,1] -> [0,1] aligns x and y when both signals are
vector-valued at each time point: x, y: [0,1] -> R^d. The C++ DP solver is
dimension-agnostic; per-channel residuals are summed into a single (N, M) D matrix
in compute_dist_matrix before the C++ call.
"""
import numpy as np
import pytest

import gdtw
from gdtw.signal import Signal
from gdtw.utils import scale

from _helpers import multid_pair, linf_phi_error, phi_true


@pytest.mark.parametrize("d", [2, 3, 5])
def test_multid_runs_and_returns_correct_shapes(d):
    t, x, y, _ = multid_pair(d=d, T=300)
    phi, x_tau, f_tau, g = gdtw.warp(x, y)
    assert g.tau.shape == (g.N,)
    assert g.D.shape == (g.N, g.M)
    assert x_tau.shape == (g.N, d)
    assert np.isfinite(f_tau)
    assert g.d == d


def test_2d_recovers_known_phi():
    t, x, y, _ = multid_pair(d=2, T=400, freqs=[5.0, 3.0])
    phi, _, f_tau, g = gdtw.warp(x, y)
    err = linf_phi_error(phi, t)
    assert err < 0.05, f"L_inf phi error = {err}"
    # f_tau scales roughly with d (sum across channels), plus regularizer terms.
    assert np.isfinite(f_tau)
    assert f_tau < 1.0


def test_2d_xtau_matches_y():
    t, x, y, _ = multid_pair(d=2, T=400, freqs=[5.0, 3.0])
    _, x_tau, _, g = gdtw.warp(x, y)
    # x(phi(t)) should match y(t) within DP-grid tolerance, per channel.
    # Use post-scaling values from g (since gdtw scales internally by default).
    y_scaled = g.y_a
    assert x_tau.shape == y_scaled.shape
    # Allow loose tolerance: high-frequency channels are sensitive to small phi error.
    assert np.max(np.abs(x_tau - y_scaled)) < 0.3


def test_identity_when_x_equals_y_3d():
    """When x == y, phi must collapse to the identity."""
    rng = np.random.default_rng(0)
    T = 300
    x = np.cumsum(rng.standard_normal((T, 3)), axis=0)
    y = x.copy()
    phi, _, f_tau, g = gdtw.warp(x, y)
    t = np.linspace(0, 1, T)
    # phi must be the identity to within DP-grid quantization.
    assert np.max(np.abs(phi(t) - t)) < 5e-2
    # f_tau won't be 0 because the DP grid samples phi at discrete points; rough random-walk
    # signals incur small interpolation loss. Still, identity warp should beat any non-trivial warp.
    _, _, f_tau_perturbed, _ = gdtw.warp(x, np.roll(y, 5, axis=0))
    assert f_tau < f_tau_perturbed


def test_per_channel_scaling_independent():
    """Channels with vastly different dynamic ranges should each end up in [-1, 1]."""
    T = 200
    t = np.linspace(0, 1, T)
    z = np.column_stack([
        1000.0 * np.sin(2 * np.pi * 2 * t),     # large channel
        0.001 * np.cos(2 * np.pi * 3 * t),      # tiny channel
    ])
    s = Signal(z, name="z", N=T, scale_signals=True)
    mins = np.nanmin(s.z_a, axis=0)
    maxs = np.nanmax(s.z_a, axis=0)
    np.testing.assert_allclose(mins, [-1.0, -1.0], atol=1e-10)
    np.testing.assert_allclose(maxs, [1.0, 1.0], atol=1e-10)


def test_scale_util_per_channel():
    """The scale() utility should treat 2-D inputs column-wise."""
    z = np.array([[0.0, 100.0], [10.0, 200.0], [20.0, 300.0]])
    out = scale(z, range=[-1, 1])
    np.testing.assert_allclose(out.min(axis=0), [-1, -1])
    np.testing.assert_allclose(out.max(axis=0), [1, 1])


def test_tuple_input_2d():
    """Array form and tuple form should produce the same phi when forced to the same grid.

    Without an explicit t, the array path infers N from x.shape[0] (=200) while the tuple
    path defaults to N_default=300, since `isinstance(tuple, ndarray)` is False. Passing
    t explicitly to both makes them apples-to-apples.
    """
    t = np.linspace(0, 1, 200)
    _, x, y, _ = multid_pair(d=2, T=200, freqs=[5.0, 3.0])
    phi_arr, _, _, _ = gdtw.warp(x, y, t=t)
    phi_tuple, _, _, _ = gdtw.warp((x, t), (y, t), t=t)
    grid = np.linspace(0, 1, 50)
    np.testing.assert_allclose(phi_arr(grid), phi_tuple(grid), atol=1e-9)


def test_l1_loss_2d_converges():
    _, x, y, _ = multid_pair(d=2, T=300, freqs=[5.0, 3.0])
    phi, _, f_tau, _ = gdtw.warp(x, y, params={"Loss": "L1"})
    assert np.isfinite(f_tau)
    t = np.linspace(0, 1, 300)
    assert linf_phi_error(phi, t) < 0.08


def test_custom_loss_matches_l2():
    """User-supplied elementwise loss should match the built-in L2 string identifier."""
    _, x, y, _ = multid_pair(d=2, T=300, freqs=[5.0, 3.0])
    _, x_tau_str, f_str, _ = gdtw.warp(x, y, params={"Loss": "L2"})
    _, x_tau_fn, f_fn, _ = gdtw.warp(x, y, params={"Loss": lambda r: r ** 2})
    assert f_str == pytest.approx(f_fn, rel=1e-9, abs=1e-12)
    np.testing.assert_allclose(x_tau_str, x_tau_fn, atol=1e-9)


def test_mismatched_channel_dim_raises():
    T = 200
    t = np.linspace(0, 1, T)
    x = np.column_stack([np.sin(t), np.cos(t)])               # d=2
    y = np.column_stack([np.sin(t), np.cos(t), np.sin(2 * t)])  # d=3
    with pytest.raises(ValueError, match=r"d=2.*d=3|d=3.*d=2"):
        gdtw.warp(x, y)


def test_callable_multid_signal():
    t = np.linspace(0, 1, 300)
    x_fn = lambda u: np.column_stack([np.sin(2 * np.pi * 5 * u), np.cos(2 * np.pi * 3 * u)])
    y_fn = lambda u: np.column_stack([np.sin(2 * np.pi * 5 * u ** 2), np.cos(2 * np.pi * 3 * u ** 2)])
    phi, _, _, _ = gdtw.warp(x_fn, y_fn, t=t)
    assert linf_phi_error(phi, t) < 0.05


def test_higher_rank_input_raises():
    """Anything beyond (T,) or (T, d) should be rejected with a clear message."""
    T = 100
    bad = np.zeros((T, 2, 2))
    good = np.zeros((T, 2))
    with pytest.raises(ValueError, match="1-D \\(T,\\) or 2-D \\(T, d\\)"):
        gdtw.warp(bad, good)
