# GDTW

_Please visit the documentation and interactive demo site at [https://dderiso.github.io/gdtw](https://dderiso.github.io/gdtw)._

GDTW is a Python/C++ library that performs dynamic time warping. 
It is based on a paper by Dave Deriso and Stephen Boyd.

## Installation

```
pip install gdtw
```

## Documentation

For full documentation, including a quick-start tutorial, please see [https://dderiso.github.io/gdtw](https://dderiso.github.io/gdtw).

## Multi-dimensional signals

Signals can be scalar `x: [0,1] -> R` (shape `(T,)`) or vector-valued `x: [0,1] -> R^d` (shape `(T, d)`). A single warping function aligns all `d` channels:

```python
import numpy as np, gdtw
t = np.linspace(0, 1, 300)
x = np.column_stack([np.sin(2*np.pi*5*t),    np.cos(2*np.pi*3*t)])
y = np.column_stack([np.sin(2*np.pi*5*t**2), np.cos(2*np.pi*3*t**2)])
phi, x_tau, f_tau, g = gdtw.warp(x, y)   # x_tau.shape == (300, 2)
```

`x` and `y` must have the same number of channels. Per-channel scaling is applied when `scale_signals=True` (the default), so channels with very different dynamic ranges are each normalized to `[-1, 1]`.

## Running the tests

From the repo root:

```
pip install -e ".[test]"
python -m pytest test/ -v
```

This runs the full suite; multi-dimensional coverage lives in
`test/test_warp.py`, which is parametrized over the channel count `d`
(plus `test/test_loss.py`, `test/test_signal.py`, `test/test_symmetric.py`).

## Numerical conventions

- The C++ dynamic program accumulates a right-endpoint quadrature of the
  paper's discretized objective, `w_0 n(0) + sum_i dt_i (e_i + n(i+1))`.
  With pinned endpoints (the default) this has the same minimizer as the
  paper's left-endpoint rule -- the boundary nodes are path-constants --
  and only the reported value differs by a constant; with relaxed
  boundaries (`BC_start_stop=False`) the initial node carries the weight
  `dt_0` like every other node.
- In the discretized program the instantaneous penalty is applied to the
  raw discrete slope, matching the paper's discretized objective.
- Symmetric mode (`symmetric=True`) reads `y` at `psi(t) = 2t - phi(t)`
  and regularizes `phi` only: `psi` is determined by the centering
  constraint, as in the paper's symmetric bidirectional formulation.
- The relaxed-boundary bounds follow the paper's extended-bounds formula
  (`beta` relaxes the two `s_max` cones only) and require
  `BC_start_stop=False`.

## Our Paper

Please see [the published article](https://rdcu.be/cT5dD).

## Citing

```
@article{deriso2022general,
  title={A general optimization framework for dynamic time warping},
  author={Deriso, Dave and Boyd, Stephen},
  journal={Optimization and Engineering},
  pages={1--22},
  year={2022},
  publisher={Springer}
}
```

## Linux

Limited Linux support. Currently supports Python 3.6 on: CentOS 7 rh-python38, CentOS 8 python38, Fedora 32+, Mageia 8+, openSUSE 15.3+, Photon OS 4.0+ (3.0+ with updates), Ubuntu 20.04+

See [manylinux](https://github.com/pypa/manylinux) for latest list of versions supported under `manylinux2014`. 
