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

From this directory (`python/`):

```
pip install -e ".[test]"
python -m pytest test/ -v
```

Or from the repo root: `pip install -e ./python && pytest python/test -v`.

Coverage:
- `test/test_warp.py` — end-to-end `warp()` behavior, parametrized over scalar (`d=1`) and multi-channel (`d=2`, `d=3`) signals: shape contracts, quadratic-phi recovery (L1/L2), identity warp, callable/tuple input routing, channel-mismatch rejection, and the pinned numerical baseline (`f_tau ≈ 0.16694595777903623`).
- `test/test_loss.py` — Huber loss / regularizer surface and custom callable losses.
- `test/test_signal.py` — `Signal` class and per-channel scaling.
- `test/test_symmetric.py` — symmetric warping path `ψ(t) = 2t − φ(t)`.

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
