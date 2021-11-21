import numpy as np
from .gdtw import GDTW

def warp(x=None, y=None, t=None, params={}):
    # directional warping: x(phi(t)) ~ y(t)
    g = GDTW().set_params(dict(params, x=x, y=y, t=t)).run()
    return g.phi, g.x_f(g.tau), g.f_tau.copy(), g