/*
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Copyright (C) 2019-2023 Dave Deriso <dderiso@alumni.stanford.edu>
 * Copyright (C) 2019-2023 Stephen Boyd
 * 
 * GDTW is a Python/C++ library that performs dynamic time warping.
 * It is based on a paper by Dave Deriso and Stephen Boyd.
 * GDTW improves upon other methods (such as the original DTW, ShapeDTW, and FastDTW) by introducing regularization, 
 * which obviates the need for pre-processing, and cross-validation for choosing optimal regularization hyper-parameters. 
 * 
 * Visit: https://github.com/dderiso/gdtw (source)
 * Visit: https://dderiso.github.io/gdtw  (docs) 
 */

#ifndef UTILS_H
#define UTILS_H

#include <Python.h>
#include <math.h>
#include <iostream>

int f_type(PyObject*& obj);

bool out_of_bounds(const double& x, const double& lower, const double& upper);

double L1(const double& x);
double L2(const double& x);

double L1(const double* x, const double* y);
double L2(const double* x, const double* y);

#endif



