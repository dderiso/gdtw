# SPDX-License-Identifier: Apache-2.0
# 
# Copyright (C) 2019-2023 Dave Deriso <dderiso@alumni.stanford.edu>, Twitter: @davederiso
# Copyright (C) 2019-2023 Stephen Boyd
# 
# GDTW is a Python/C++ library that performs dynamic time warping.
# GDTW improves upon other methods (such as the original DTW, ShapeDTW, and FastDTW) by introducing regularization, 
# which obviates the need for pre-processing, and cross-validation for choosing optimal regularization hyper-parameters. 
# 
# Paper: https://rdcu.be/cT5dD
# Source: https://github.com/dderiso/gdtw
# Docs: https://dderiso.github.io/gdtw


import numpy as np
from .gdtw import GDTW

def warp(x=None, y=None, t=None, params={}):
    # directional warping: x(phi(t)) ~ y(t)
    g = GDTW().set_params(dict(params, x=x, y=y, t=t)).run()
    return g.phi, g.x_f(g.tau), g.f_tau.copy(), g