import numpy as np


def phi_true(t):
    return t ** 2


def scalar_pair(T=200, freq=5.0):
    """Scalar (T,) sinusoidal pair with y(t) = x(phi_true(t))."""
    t = np.linspace(0, 1, T)
    x = np.sin(2 * np.pi * freq * t)
    y = np.sin(2 * np.pi * freq * phi_true(t))
    return t, x, y


def multid_pair(d, T=300, freqs=None, seed=0):
    """
    Build a (T, d) signal pair x, y where y(t) = x(phi_true(t)).
    Each channel is a sinusoid with a distinct frequency.
    """
    if freqs is None:
        rng = np.random.default_rng(seed)
        freqs = 2.0 + rng.uniform(0.0, 6.0, size=d)
    t = np.linspace(0, 1, T)
    x = np.column_stack([np.sin(2 * np.pi * f * t) for f in freqs])
    y = np.column_stack([np.sin(2 * np.pi * f * phi_true(t)) for f in freqs])
    return t, x, y, freqs


def make_pair(d, T=300, freqs=None):
    """Dispatch to scalar_pair (d=1) or multid_pair (d>=2). Returns (t, x, y)."""
    if d == 1:
        return scalar_pair(T=T)
    t, x, y, _ = multid_pair(d=d, T=T, freqs=freqs)
    return t, x, y


def linf_phi_error(phi_hat, t):
    return float(np.max(np.abs(phi_hat(t) - phi_true(t))))
