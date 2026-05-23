"""
Signal-class and scale() utility unit tests.

These exercise the preprocessing layer in gdtw.signal / gdtw.utils directly,
without routing through warp(). End-to-end behavior lives in test_warp.py.
"""
import numpy as np

from gdtw.signal import Signal
from gdtw.utils import scale


def test_signal_per_channel_scaling_independent():
    """Channels with vastly different dynamic ranges should each end up in [-1, 1]."""
    T = 200
    t = np.linspace(0, 1, T)
    z = np.column_stack([
        1000.0 * np.sin(2 * np.pi * 2 * t),     # large channel
        0.001 * np.cos(2 * np.pi * 3 * t),      # tiny channel
    ])
    s = Signal(z, name="z", N=T, scale_signals=True)
    np.testing.assert_allclose(np.nanmin(s.z_a, axis=0), [-1.0, -1.0], atol=1e-10)
    np.testing.assert_allclose(np.nanmax(s.z_a, axis=0), [1.0, 1.0], atol=1e-10)


def test_scale_util_per_channel():
    """The scale() utility should treat 2-D inputs column-wise."""
    z = np.array([[0.0, 100.0], [10.0, 200.0], [20.0, 300.0]])
    out = scale(z, range=[-1, 1])
    np.testing.assert_allclose(out.min(axis=0), [-1, -1])
    np.testing.assert_allclose(out.max(axis=0), [1, 1])
