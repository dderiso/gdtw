# gdtw (C++ port)

An all-C++ port of the Python `gdtw` package living next to it. Same algorithm, same defaults, same tests.

## Build

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Test

```
ctest --test-dir build --output-on-failure
```

## Layout

```
include/gdtw/   public headers (types, utils, signal, solver, gdtw, warp)
src/            implementation
tests/          one-for-one C++ port of python/test/ (doctest)
third_party/    vendored doctest.h
```

## Parity with the Python package

The test in `tests/test_warp.cpp` named `test_warp_baseline_pinned` asserts the same `f_tau` (`0.16694595777903623`) and `phi(0.5)` (`0.25318182065`) as the Python regression anchor. If those drift, the port is wrong.
