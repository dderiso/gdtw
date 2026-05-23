# GDTW

_Documentation and interactive demo: [https://dderiso.github.io/gdtw](https://dderiso.github.io/gdtw)._

GDTW is a library for **general dynamic time warping** — DTW with regularization that obviates pre-processing and cross-validation for hyper-parameter selection.

## Paper

Deriso, D. & Boyd, S. *A general optimization framework for dynamic time warping.* Optimization and Engineering, 2022. [rdcu.be/cT5dD](https://rdcu.be/cT5dD)

---

### Implementations

|                          | `python/`                              | `cpp/`                                |
| ---                      | ---                                    | ---                                   |
| Core class               | `gdtw.GDTW`                            | `gdtw::GDTW`                          |
| DP kernel                | `gdtw/solver.hpp`                      | `cpp/src/solver.cpp`                  |
| Build system             | `setuptools`                           | CMake 3.20+, C++20                    |
| External deps            | NumPy                                  | none                                  |
| Test framework           | pytest                                 | doctest                               |
| Multi-channel signals    | ✓                                      | ✓                                     |
| Symmetric mode           | ✓                                      | ✓                                     |
| Losses / regularizers    | L1, L2, Huber, custom                  | L1, L2, Huber, custom                 |

### Parity anchor

| Metric        | Python (`-Ofast`)         | C++ (`-O2`)               | Δ                |
| ---           | ---                       | ---                       | ---              |
| `f_tau`       | `0.166945957779036230`    | `0.166945957779036175`    | ~0.25 ULP        |
| `phi(0.5)`    | `0.253181820650000000`    | `0.253181820650000000`    | 0                |

Same DP recurrence on both sides; the `f_tau` gap is FP non-associativity in the inner sum from different compiler reduction order, at the precision floor for a cross-build.

## License

Apache-2.0 — see `LICENSE`.
