# GDTW

_Documentation and interactive demo: [https://dderiso.github.io/gdtw](https://dderiso.github.io/gdtw)._

GDTW is a library for **general dynamic time warping** — DTW with regularization that obviates pre-processing and cross-validation for hyper-parameter selection.

## Paper

Deriso, D. & Boyd, S. *A general optimization framework for dynamic time warping.* Optimization and Engineering, 2022. [rdcu.be/cT5dD](https://rdcu.be/cT5dD)

---

This repo has two parallel implementations of the same algorithm:

```
gdtw/
├── python/   The released pip package. Python orchestration + small C++ DP kernel
│            built via a hand-rolled CPython/NumPy extension (`gdtwcpp.solve`).
│            Install:  pip install ./python
│            Test:     pytest python/test -v
│
└── cpp/      All-C++ port: same algorithm, same defaults, same tests. No Python.
             Build:    cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
                       cmake --build cpp/build -j
             Test:     ctest --test-dir cpp/build --output-on-failure
```

The two implementations share the same iterative DP solver and the same test surface; `cpp/tests/` is a one-for-one port of `python/test/`. The pinned baseline in `test_warp_baseline_pinned` is the parity anchor between the two.

## At a glance

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
