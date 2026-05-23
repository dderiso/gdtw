# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2017-2026 
# Dave Deriso <dderiso@alumni.stanford.edu>, Twitter: @davederiso
# Stephen Boyd
#
# GDTW is a Python/C++ library that performs dynamic time warping.
# GDTW improves upon other methods (such as the original DTW, ShapeDTW, and FastDTW) by introducing regularization,
# which obviates the need for pre-processing, and cross-validation for choosing optimal regularization hyper-parameters.
#
# Paper: https://rdcu.be/cT5dD
# Source: https://github.com/dderiso/gdtw
# Docs: https://dderiso.github.io/gdtw

"""
Symmetric warping path: ψ(t) = 2t − φ(t) computed inside the same DP solve.

Asymmetric mode reads y at t_i and varies x at Tau[i, j]. Symmetric mode keeps
the same DP, but reads y at ψ(t_i, j) = 2 t_i − Tau[i, j] so the residual
couples both directions.

End-to-end shape and identity-warp behavior is covered by parametrized cases
in test_warp.py; this file pins the symmetric-only invariants (ψ accessor,
ψ + φ = 2t, default-is-asymmetric).
"""
import numpy as np
import pytest

import gdtw

from _helpers import scalar_pair


def test_symmetric_exposes_psi_accessors():
    _, x, y = scalar_pair(T=200)
    _, _, _, g = gdtw.warp(x, y, params={"symmetric": True})
    grid = np.linspace(0, 1, 50)
    np.testing.assert_allclose(g.psi(grid), 2.0 * grid - g.phi(grid), atol=1e-12)
    assert g.get_psi_values().shape == (g.N,)


def test_symmetric_psi_complementary():
    """ψ(t) + φ(t) == 2t at every node, by construction."""
    _, x, y = scalar_pair(T=200)
    _, _, _, g = gdtw.warp(x, y, params={"symmetric": True})
    np.testing.assert_allclose(g.get_psi_values() + g.tau, 2.0 * g.t, atol=1e-12)


def test_symmetric_default_is_asymmetric():
    """Omitting `symmetric` must reproduce the legacy asymmetric path bit-identically."""
    _, x, y = scalar_pair(T=200)
    _, _, f_default, _ = gdtw.warp(x, y)
    _, _, f_explicit, _ = gdtw.warp(x, y, params={"symmetric": False})
    assert f_default == pytest.approx(f_explicit, rel=1e-12, abs=1e-12)
