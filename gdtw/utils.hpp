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



