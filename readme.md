# GDTW

_Documentation and interactive demo: [https://dderiso.github.io/gdtw](https://dderiso.github.io/gdtw)._

GDTW is a library for **general dynamic time warping** — DTW with regularization that obviates pre-processing and cross-validation for hyper-parameter selection. Based on a paper by Dave Deriso and Stephen Boyd ([rdcu.be/cT5dD](https://rdcu.be/cT5dD)).

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

The two implementations share the same iterative DP solver and the same test surface; `cpp/tests/` is a one-for-one port of `python/test/`. The pinned baseline in `test_warp_baseline_pinned` (`f_tau ≈ 0.16694595777903623`) is the parity anchor between the two.

## Paper

Deriso, D. & Boyd, S. *A general optimization framework for dynamic time warping.* Optimization and Engineering, 2022. [rdcu.be/cT5dD](https://rdcu.be/cT5dD)

## License

Apache-2.0 — see `LICENSE`.
